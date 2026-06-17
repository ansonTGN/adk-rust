mod cli;
mod deploy;
mod graph;
mod setup;
mod skills;
mod ultra;

use adk_agent::LlmAgentBuilder;
use adk_agent::coding::CodingAgent;
use adk_cli::{Launcher, launcher::ThinkingDisplayMode};
use adk_core::{Content, Llm, Part, SessionId, UserId};
use adk_devtools::Workspace;
use adk_model::ModelProvider;
use adk_runner::Runner;
use adk_session::{CreateRequest, InMemorySessionService, SessionService};
use adk_tool::GoogleSearchTool;
use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, ThinkingMode};
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Chat) => {
            let agent = build_agent(
                cli.provider,
                cli.model,
                cli.api_key,
                cli.instruction,
                cli.thinking_budget,
            )?;
            Launcher::new(Arc::new(agent))
                .app_name("adk-rust")
                .with_thinking_mode(map_thinking_mode(cli.thinking_mode))
                .run_console_directly()
                .await
                .map_err(Into::into)
        }
        Some(Commands::Serve { port }) => {
            let agent = build_agent(
                cli.provider,
                cli.model,
                cli.api_key,
                cli.instruction,
                cli.thinking_budget,
            )?;
            Launcher::new(Arc::new(agent))
                .app_name("adk-rust")
                .run_serve_directly(port)
                .await
                .map_err(Into::into)
        }
        Some(Commands::Code { task, dir, read_only }) => {
            run_code(
                cli.provider,
                cli.model,
                cli.api_key,
                cli.thinking_budget,
                dir,
                read_only,
                task,
            )
            .await
        }
        Some(Commands::Goal { goal, until, dir, max_iters }) => {
            run_goal(
                cli.provider,
                cli.model,
                cli.api_key,
                cli.thinking_budget,
                dir,
                goal,
                until,
                max_iters,
            )
            .await
        }
        Some(Commands::Ultracode { task, dir, max_rounds }) => {
            ultra::run(
                cli.provider,
                cli.model,
                cli.api_key,
                cli.thinking_budget,
                dir,
                task,
                max_rounds,
            )
            .await
        }
        Some(Commands::Skills { command }) => skills::run(command),
        Some(Commands::Deploy { command }) => deploy::run(command).await,
        Some(Commands::Graph { command }) => graph::run(command).await,
    }
}

/// Resolve provider/model/key non-interactively (default: a Gemini 3 model;
/// key from `--api-key` or the environment). Returns the model and its id.
fn resolve_model(
    cli_provider: Option<ModelProvider>,
    cli_model: Option<String>,
    cli_api_key: Option<String>,
    thinking_budget: Option<u32>,
) -> Result<(Arc<dyn Llm>, String)> {
    let provider = cli_provider.unwrap_or(ModelProvider::Gemini);
    let model_id = cli_model.unwrap_or_else(|| match provider {
        ModelProvider::Gemini => "gemini-3.1-flash-lite".to_string(),
        _ => provider.default_model().to_string(),
    });
    let api_key = cli_api_key.or_else(|| env_api_key(provider));
    let model = create_model(provider, &model_id, api_key.as_deref(), thinking_budget)?;
    Ok((model, model_id))
}

/// Drive one agent turn on an existing runner/session, streaming the trace.
async fn stream_turn(runner: &Runner, session_id: &str, prompt: &str) -> Result<()> {
    let mut stream = runner
        .run(
            UserId::new("user")?,
            SessionId::new(session_id)?,
            Content::new("user").with_text(prompt),
        )
        .await?;
    let mut pending = String::new();
    while let Some(event) = stream.next().await {
        let event = event?;
        if let Some(content) = &event.llm_response.content {
            for part in &content.parts {
                match part {
                    Part::FunctionCall { name, args, .. } => {
                        flush_text(&mut pending);
                        println!("  🔧 {name}({})", first_line(&args.to_string()));
                    }
                    Part::FunctionResponse { function_response, .. } => {
                        flush_text(&mut pending);
                        println!("  ↩  {}", first_line(&function_response.response.to_string()));
                    }
                    Part::Text { text } if !text.is_empty() => pending.push_str(text),
                    _ => {}
                }
            }
        }
    }
    flush_text(&mut pending);
    Ok(())
}

/// Run a shell command in `dir`; returns (exit_code, combined stdout+stderr).
fn run_check(dir: &str, command: &str) -> (Option<i32>, String) {
    match std::process::Command::new("sh").arg("-c").arg(command).current_dir(dir).output() {
        Ok(o) => {
            let mut out = String::from_utf8_lossy(&o.stdout).to_string();
            out.push_str(&String::from_utf8_lossy(&o.stderr));
            (o.status.code(), out)
        }
        Err(e) => (None, e.to_string()),
    }
}

/// Autonomous goal mode: loop plan → act → verify until `until` passes or budget.
#[allow(clippy::too_many_arguments)]
async fn run_goal(
    cli_provider: Option<ModelProvider>,
    cli_model: Option<String>,
    cli_api_key: Option<String>,
    thinking_budget: Option<u32>,
    dir: String,
    goal: String,
    until: String,
    max_iters: u32,
) -> Result<()> {
    let (model, model_id) = resolve_model(cli_provider, cli_model, cli_api_key, thinking_budget)?;
    let coding = CodingAgent::builder().model(model).workspace(Workspace::new(&dir)).build()?;

    // One runner/session: the agent remembers prior attempts across iterations.
    let sessions: Arc<dyn SessionService> = Arc::new(InMemorySessionService::new());
    sessions
        .create(CreateRequest {
            app_name: "adk-rust".into(),
            user_id: "user".into(),
            session_id: Some("goal".into()),
            state: Default::default(),
        })
        .await?;
    let runner = Runner::builder()
        .app_name("adk-rust")
        .agent(coding.agent())
        .session_service(sessions)
        .build()?;

    println!("goal mode ({model_id}) on {dir}");
    println!("goal:  {goal}");
    println!("until: {until}  (budget: {max_iters} iterations)\n");

    let mut last_output = String::new();
    for iter in 1..=max_iters {
        println!("━━ iteration {iter}/{max_iters} ━━");
        let prompt = if iter == 1 {
            format!(
                "Goal: {goal}\n\nWhen you believe the goal is met, the command `{until}` must \
                 succeed (exit 0). Work toward that now."
            )
        } else {
            format!(
                "The success check `{until}` is not passing yet. Its latest output was:\n\
                 ---\n{last_output}\n---\nDiagnose and fix this, continuing toward the goal: {goal}"
            )
        };
        stream_turn(&runner, "goal", &prompt).await?;

        let (code, output) = run_check(&dir, &until);
        last_output = first_lines(&output, 40);
        if code == Some(0) {
            println!("\n✅ goal met after {iter} iteration(s): `{until}` passed.");
            return Ok(());
        }
        println!("  ✗ check `{until}` exited {code:?}; iterating.\n");
    }
    println!("\n⚠️  budget exhausted after {max_iters} iteration(s); `{until}` still failing.");
    std::process::exit(1);
}

/// First `n` lines of `s` (for compact failure feedback).
fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join("\n")
}

/// Run the coding agent on a single task in a workspace directory.
#[allow(clippy::too_many_arguments)]
async fn run_code(
    cli_provider: Option<ModelProvider>,
    cli_model: Option<String>,
    cli_api_key: Option<String>,
    thinking_budget: Option<u32>,
    dir: String,
    read_only: bool,
    task: String,
) -> Result<()> {
    let (model, model_id) = resolve_model(cli_provider, cli_model, cli_api_key, thinking_budget)?;
    let workspace = if read_only { Workspace::read_only(&dir) } else { Workspace::new(&dir) };
    let coding = CodingAgent::builder().model(model).workspace(workspace).build()?;

    let sessions: Arc<dyn SessionService> = Arc::new(InMemorySessionService::new());
    sessions
        .create(CreateRequest {
            app_name: "adk-rust".into(),
            user_id: "user".into(),
            session_id: Some("code".into()),
            state: Default::default(),
        })
        .await?;
    let runner = Runner::builder()
        .app_name("adk-rust")
        .agent(coding.agent())
        .session_service(sessions)
        .build()?;

    println!("coding agent ({model_id}) on {dir}\ntask: {task}\n");
    stream_turn(&runner, "code", &task).await?;

    let todos = coding.todos();
    if !todos.is_empty() {
        println!("\nplan:");
        for t in todos {
            let mark = match t.status.as_str() {
                "completed" => "✓",
                "in_progress" => "▶",
                _ => "·",
            };
            println!("  {mark} {}", t.content);
        }
    }
    Ok(())
}

/// Read the API key for a provider from its conventional environment variable.
fn env_api_key(provider: ModelProvider) -> Option<String> {
    let try_vars: &[&str] = match provider {
        ModelProvider::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        ModelProvider::Openai => &["OPENAI_API_KEY"],
        ModelProvider::Anthropic => &["ANTHROPIC_API_KEY"],
        ModelProvider::Deepseek => &["DEEPSEEK_API_KEY"],
        ModelProvider::Groq => &["GROQ_API_KEY"],
        ModelProvider::Ollama => &[],
    };
    try_vars.iter().find_map(|v| std::env::var(v).ok())
}

fn flush_text(pending: &mut String) {
    let trimmed = pending.trim();
    if !trimmed.is_empty() {
        println!("  🤖 {trimmed}");
    }
    pending.clear();
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.len() > 160 { format!("{}…", &line[..160]) } else { line.to_string() }
}

fn build_agent(
    cli_provider: Option<ModelProvider>,
    cli_model: Option<String>,
    cli_api_key: Option<String>,
    cli_instruction: Option<String>,
    thinking_budget: Option<u32>,
) -> Result<adk_agent::LlmAgent> {
    let resolved = setup::resolve(cli_provider, cli_model, cli_api_key, cli_instruction)?;
    let model = create_model(
        resolved.provider,
        &resolved.model,
        resolved.api_key.as_deref(),
        thinking_budget,
    )?;

    let mut builder = LlmAgentBuilder::new("adk_agent")
        .description("Default ADK-Rust CLI agent")
        .instruction(resolved.instruction)
        .model(model);

    // Google Search grounding only works with Gemini
    if resolved.provider == ModelProvider::Gemini {
        builder = builder.tool(Arc::new(GoogleSearchTool::new()));
    }

    builder.build().map_err(Into::into)
}

fn create_model(
    provider: ModelProvider,
    model: &str,
    api_key: Option<&str>,
    thinking_budget: Option<u32>,
) -> Result<Arc<dyn Llm>> {
    match provider {
        #[cfg(feature = "gemini")]
        ModelProvider::Gemini => {
            reject_unsupported_thinking_budget(provider, thinking_budget)?;
            let key = api_key.ok_or_else(|| anyhow::anyhow!("Gemini requires an API key"))?;
            let m = adk_model::GeminiModel::new(key, model)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "gemini"))]
        ModelProvider::Gemini => provider_feature_disabled(provider, "gemini"),
        #[cfg(feature = "openai")]
        ModelProvider::Openai => {
            reject_unsupported_thinking_budget(provider, thinking_budget)?;
            let key = api_key.ok_or_else(|| anyhow::anyhow!("OpenAI requires an API key"))?;
            let config = adk_model::OpenAIConfig::new(key, model);
            let m = adk_model::OpenAIClient::new(config)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "openai"))]
        ModelProvider::Openai => provider_feature_disabled(provider, "openai"),
        #[cfg(feature = "anthropic")]
        ModelProvider::Anthropic => {
            let key = api_key.ok_or_else(|| anyhow::anyhow!("Anthropic requires an API key"))?;
            let mut config = adk_model::anthropic::AnthropicConfig::new(key, model);
            if let Some(budget) = thinking_budget {
                if budget == 0 {
                    return Err(anyhow::anyhow!("--thinking-budget must be greater than 0"));
                }
                config = config.with_thinking(budget);
            }
            let m = adk_model::AnthropicClient::new(config)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "anthropic"))]
        ModelProvider::Anthropic => provider_feature_disabled(provider, "anthropic"),
        #[cfg(feature = "deepseek")]
        ModelProvider::Deepseek => {
            reject_unsupported_thinking_budget(provider, thinking_budget)?;
            let key = api_key.ok_or_else(|| anyhow::anyhow!("DeepSeek requires an API key"))?;
            let config = adk_model::DeepSeekConfig::new(key, model);
            let m = adk_model::DeepSeekClient::new(config)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "deepseek"))]
        ModelProvider::Deepseek => provider_feature_disabled(provider, "deepseek"),
        #[cfg(feature = "groq")]
        ModelProvider::Groq => {
            reject_unsupported_thinking_budget(provider, thinking_budget)?;
            let key = api_key.ok_or_else(|| anyhow::anyhow!("Groq requires an API key"))?;
            let config = adk_model::GroqConfig::new(key, model);
            let m = adk_model::GroqClient::new(config)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "groq"))]
        ModelProvider::Groq => provider_feature_disabled(provider, "groq"),
        #[cfg(feature = "ollama")]
        ModelProvider::Ollama => {
            reject_unsupported_thinking_budget(provider, thinking_budget)?;
            let config = adk_model::OllamaConfig::new(model);
            let m = adk_model::OllamaModel::new(config)?;
            Ok(Arc::new(m))
        }
        #[cfg(not(feature = "ollama"))]
        ModelProvider::Ollama => provider_feature_disabled(provider, "ollama"),
    }
}

fn provider_feature_disabled(provider: ModelProvider, feature: &str) -> Result<Arc<dyn Llm>> {
    Err(anyhow::anyhow!(
        "{} support is not compiled into this adk-cli build. Reinstall with `--features {}` or `--features all-providers`.",
        provider.display_name(),
        feature
    ))
}

fn reject_unsupported_thinking_budget(
    provider: ModelProvider,
    thinking_budget: Option<u32>,
) -> Result<()> {
    if thinking_budget.is_some() {
        Err(anyhow::anyhow!("--thinking-budget is not supported for provider {}", provider))
    } else {
        Ok(())
    }
}

fn map_thinking_mode(mode: ThinkingMode) -> ThinkingDisplayMode {
    match mode {
        ThinkingMode::Auto => ThinkingDisplayMode::Auto,
        ThinkingMode::Show => ThinkingDisplayMode::Show,
        ThinkingMode::Hide => ThinkingDisplayMode::Hide,
    }
}
