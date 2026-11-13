# Assignment 35 — Text-to-Math Problem Solver Agent

A LangChain agent that translates arithmetic word problems into tool-verified calculations, solves simple one-variable linear equations, explains its steps, and keeps conversation context in a Streamlit chat app.

## Assignment overview

1. **What is a text-to-math problem?** A natural-language question that describes quantities and relationships which must be translated into mathematical operations or equations before solving.
2. **Why use an agent for math reasoning?** It can decide which calculation is needed, call a calculator or equation-solving tool, inspect the result, and then explain a grounded answer.
3. **LLM response vs. agent reasoning:** A plain LLM produces a response directly. An agent coordinates the model with tools in a choose–act–observe loop, so numeric results can be computed and checked instead of merely generated.

## Features

- LangChain agent built with `create_agent`.
- Safe calculator tool for arithmetic and percentage calculations; it parses a restricted Python AST and never evaluates arbitrary code.
- Linear equation tool for equations in `x`, such as `2*x + 3 = 11`.
- Ollama, OpenAI, and Hugging Face model providers.
- Streamlit chat UI with `st.session_state` history and a unique LangGraph checkpoint thread per browser session.
- CLI mode for a one-off problem or a persistent interactive session.
- Offline tests use a deterministic fake chat model and make no external model calls.

## Setup (Windows PowerShell)

Use Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set up one provider:

### Ollama (local)

Install Ollama, then download a model:

```powershell
ollama pull llama3.1
```

The default provider is Ollama at `http://localhost:11434`. To use another model or endpoint, edit `.env`.

### OpenAI

Set `MATH_AGENT_PROVIDER=openai` and add your key to `.env`:

```text
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

### Hugging Face

Set `MATH_AGENT_PROVIDER=huggingface`, provide `HUGGINGFACEHUB_API_TOKEN` (or `HF_TOKEN`), and select a chat model that supports tool calling in `HF_MODEL`.

Never commit `.env` or paste API keys into source files. `.env` is ignored by Git.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Example questions:

- `I have 3 boxes with 4 apples in each. How many apples do I have?`
- `What is 15% of 80?`
- `Solve 2*x + 3 = 11.`

The app stores displayed user and assistant turns in `st.session_state`. LangGraph's in-memory checkpointer stores the model/tool message state under a unique thread ID, so follow-up questions retain conversation context. Clearing the conversation or changing providers starts a fresh thread.

## Run from the command line

Single question:

```powershell
python main.py --provider ollama --question "What is 15% of 80?"
```

Interactive session (conversation context is retained until exit):

```powershell
python main.py --provider ollama
```

Use `--provider openai` or `--provider huggingface` after configuring the corresponding credentials.

## How it works

1. The agent receives the question and its system instructions.
2. It identifies the needed operation and calls the calculator or linear-equation tool.
3. The tool validates and computes the result.
4. The model explains the setup, calculation, and final answer.
5. In Streamlit, the exchange is displayed from session history and the agent's LangGraph checkpoint preserves the full message context for the next turn.

### Tool limits

- The calculator supports numeric literals, parentheses, unary signs, and `+`, `-`, `*`, `/`, `//`, `%`, and bounded integer powers. It rejects names, function calls, and other Python syntax.
- The equation solver supports linear expressions containing `x`, numbers, and basic arithmetic. It reports no-solution and infinitely-many-solutions cases.
- Word-problem interpretation and natural-language explanations come from the selected LLM; calculations are verified by the tools.

## Tests

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests cover arithmetic, percentages, simple algebra, invalid/nonlinear input, LangChain tool use, and retained session context. They require no API key or running model.
