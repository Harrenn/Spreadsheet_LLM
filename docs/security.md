# Security and data handling

- Keep real spreadsheets and downloaded models under ignored `working/` directories.
- Read Google, Gemini, and Hugging Face credentials from environment variables or runtime prompts only.
- Never save credentials in notebook cells or outputs.
- Clear notebook execution state before sharing.
- Remember that external APIs receive the address text sent for processing.
- Review whether operational data is permitted to leave the local environment before selecting an API provider.
- Local GGUF parsing avoids sending row text to an LLM API but still requires careful model/source review and adequate hardware.
