# API Key Setup Guide

This guide covers setting up API keys for enhanced Helix functionality, particularly the LLM judge component for rubric-based evaluation of AF partition analysis.

## Overview

The Helix LLM judge provides intelligent evaluation of operator-algebraic responses using large language models. While the system works without API keys (using heuristic fallbacks), connecting an LLM API significantly enhances the quality of evaluations.

## Supported API Providers

### OpenAI (Recommended)

OpenAI provides robust and reliable language models suitable for mathematical and scientific evaluation.

**Supported Models:**
- `gpt-4o-mini` (recommended, cost-effective)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Setup Steps

#### 1. Obtain OpenAI API Key

1. Visit [OpenAI API Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-...`)

**Important:** Store your API key securely and never commit it to version control.

#### 2. Configure Environment Variable

The recommended approach is to set the API key as an environment variable:

##### On Linux/macOS:
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Or set for current session only
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

##### On Windows:
```cmd
# Command Prompt
set OPENAI_API_KEY=sk-your-actual-api-key-here

# PowerShell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"

# Permanent setting (System Properties > Environment Variables)
```

#### 3. Verify Setup

Test your API key configuration:

```python
import os
from environments.helixenv.llm_judge import LLMJudge

# Check if API key is available
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print("✓ API key found")

    # Test LLM judge initialization
    judge = LLMJudge(api_key=api_key)
    print(f"✓ LLM judge initialized with model: {judge.model}")
else:
    print("✗ No API key found - using fallback mode")
```

## Usage Examples

### Basic LLM Judge Usage

```python
from environments.helixenv.llm_judge import LLMJudge, CalibrationExample

# Create calibration examples for physics-aware evaluation
calibration_examples = [
    CalibrationExample(
        prompt="What is the spectral gap in operator theory?",
        response="The spectral gap is the difference between the largest and second-largest eigenvalue magnitudes of an operator, indicating mixing speed.",
        physics_score=0.9,
        reasoning="Precise technical definition with correct physical interpretation",
        key_concepts=["spectral gap", "eigenvalues", "mixing", "operator theory"]
    )
]

# Initialize judge with API key (automatic from environment)
judge = LLMJudge(calibration_examples=calibration_examples)

# Evaluate a response
result = judge.evaluate(
    prompt="Explain the role of spectral gaps in Markov chain analysis",
    response="Spectral gaps determine how quickly Markov chains converge to equilibrium"
)

print(f"Physics Score: {result['physics_score']:.2f}")
print(f"Reasoning: {result['reasoning']}")
print(f"Key Concepts: {result['key_concepts']}")
```

### Helixenv Integration

```python
from environments.helixenv import load_environment

# Load environment with LLM judge enabled
env = load_environment(
    enable_llm_judge=True,
    llm_judge_model="gpt-4o-mini",
    samples=1024,
    width=24,
    epochs=50
)

# The environment will automatically use the API key from environment variables
# and provide enhanced evaluation of AF partition responses
```

### CLI Usage

```bash
# Run helixenv with LLM judge (uses OPENAI_API_KEY automatically)
./helix helixenv --enable-llm-judge --samples 1024 --width 24

# Specify custom model
./helix helixenv --enable-llm-judge --llm-judge-model gpt-4o --samples 512
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None (uses fallback) |
| `OPENAI_BASE_URL` | Custom API base URL | `https://api.openai.com/v1` |
| `OPENAI_ORG_ID` | Organization ID (optional) | None |

### LLM Judge Parameters

```python
judge = LLMJudge(
    api_key=None,                          # Auto-detect from environment
    model="gpt-4o-mini",                   # Model to use
    base_url="https://api.openai.com/v1",  # API endpoint
    timeout=30,                            # Request timeout
    max_retries=3,                         # Retry failed requests
    calibration_examples=examples          # Physics-aware calibration
)
```

## Cost Management

### Model Cost Comparison (as of 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Recommended Use |
|-------|----------------------|------------------------|-----------------|
| gpt-4o-mini | $0.15 | $0.60 | General evaluation (recommended) |
| gpt-4o | $5.00 | $15.00 | High-precision evaluation |
| gpt-4-turbo | $10.00 | $30.00 | Research applications |

### Cost Optimization Tips

1. **Use gpt-4o-mini** for most evaluations (90% cost reduction vs gpt-4)
2. **Batch evaluations** when possible
3. **Set usage limits** in OpenAI dashboard
4. **Monitor usage** through OpenAI usage dashboard
5. **Use fallback mode** for development/testing

### Usage Estimation

Typical evaluation:
- **Prompt length**: ~500 tokens (including calibration examples)
- **Response length**: ~200 tokens
- **LLM output**: ~100 tokens
- **Total per evaluation**: ~800 tokens

Cost per 1000 evaluations with gpt-4o-mini: ~$0.50

## Troubleshooting

### Common Issues

#### "Invalid API Key" Error
```
openai.AuthenticationError: Incorrect API key provided
```

**Solutions:**
1. Verify API key is correctly set: `echo $OPENAI_API_KEY`
2. Check for extra spaces or newlines in the key
3. Ensure key hasn't expired or been revoked
4. Generate a new key from OpenAI dashboard

#### "Rate Limit Exceeded" Error
```
openai.RateLimitError: Rate limit reached
```

**Solutions:**
1. Implement exponential backoff (built into LLMJudge)
2. Reduce evaluation frequency
3. Upgrade to higher rate limit tier
4. Use fallback mode temporarily

#### "Model Not Found" Error
```
openai.NotFoundError: Model 'gpt-4' not found
```

**Solutions:**
1. Check model name spelling
2. Verify account has access to the model
3. Use a different model (e.g., `gpt-4o-mini`)

#### Network/Timeout Errors
```
openai.APITimeoutError: Request timed out
```

**Solutions:**
1. Check internet connection
2. Increase timeout parameter
3. Retry the request
4. Use fallback mode if persistent

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# LLM judge will now provide detailed API call information
judge = LLMJudge(api_key="your-key")
```

### Fallback Behavior

When API calls fail, the LLM judge automatically falls back to heuristic evaluation:

```python
# This will work even without valid API key
judge = LLMJudge(api_key=None)  # or invalid key

result = judge.evaluate("prompt", "response")
# Returns heuristic evaluation with physics keyword matching
```

## Security Best Practices

### API Key Security

1. **Never commit API keys** to version control
2. **Use environment variables** for key storage
3. **Rotate keys regularly** (every 90 days)
4. **Set usage limits** in OpenAI dashboard
5. **Monitor usage** for unexpected activity

### Key Storage

#### ✅ Secure Methods:
- Environment variables
- Secure key management systems (AWS Secrets Manager, etc.)
- Encrypted configuration files

#### ❌ Insecure Methods:
- Hardcoded in source code
- Plain text configuration files
- Shared in chat/email

### Example Secure Setup

```bash
# Create a secure environment setup script
cat > setup_api_keys.sh << 'EOF'
#!/bin/bash
# Secure API key setup for Helix

# Check if key file exists
if [[ -f ~/.helix_api_key ]]; then
    export OPENAI_API_KEY=$(cat ~/.helix_api_key)
    echo "✓ API key loaded from secure file"
else
    echo "⚠ No API key file found. Creating ~/.helix_api_key"
    echo "Please enter your OpenAI API key:"
    read -s api_key
    echo "$api_key" > ~/.helix_api_key
    chmod 600 ~/.helix_api_key  # Restrict file permissions
    export OPENAI_API_KEY="$api_key"
    echo "✓ API key saved securely"
fi
EOF

chmod +x setup_api_keys.sh
source setup_api_keys.sh
```

## Support

### Getting Help

1. **Documentation**: Check this guide and API documentation
2. **Environment variables**: Verify using `env | grep OPENAI`
3. **OpenAI Status**: Check [status.openai.com](https://status.openai.com)
4. **Helix Issues**: Report issues at the project repository

### Reporting Issues

When reporting API-related issues, please include:

1. Operating system and Python version
2. Helix version
3. Error messages (redact API keys!)
4. Steps to reproduce
5. Whether fallback mode works

**Example Issue Report:**
```
Environment: macOS 14.0, Python 3.11
Helix Version: [current version]
Issue: LLM judge fails with timeout error
Error: openai.APITimeoutError: Request timed out
Fallback works: Yes
Steps: Called judge.evaluate() with long prompt
```

---

For additional support or questions about API integration, please refer to the main project documentation or open an issue in the repository.