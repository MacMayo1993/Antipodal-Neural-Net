# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by emailing the maintainers at:

**macmayo1993@users.noreply.github.com**

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)

We will respond to security reports within **48 hours** and will work with you to address the issue promptly.

## Security Best Practices

When using this software:

1. **Dependencies**: Regularly update dependencies to their latest secure versions
   ```bash
   pip install --upgrade torch numpy pytest
   ```

2. **Model Checkpoints**: Only load model checkpoints from trusted sources
   - Verify checksums before loading
   - Use `torch.load(..., weights_only=True)` when loading untrusted checkpoints

3. **Data**: Validate and sanitize input data
   - Check tensor shapes and dtypes
   - Handle NaN/Inf values appropriately
   - Limit input sequence lengths to prevent memory exhaustion

4. **Execution**: Run benchmarks and experiments in isolated environments
   - Use virtual environments or containers
   - Limit computational resources for untrusted code

## Known Security Considerations

- **Torch Model Loading**: PyTorch's `torch.load()` uses pickle, which can execute arbitrary code. Only load models from trusted sources.
- **Resource Limits**: Long-running benchmarks may exhaust system resources. Use `--steps` and `--seeds` flags to control execution time.

## Disclosure Policy

We follow responsible disclosure practices:
- Security issues are addressed privately before public disclosure
- Credit is given to researchers who report valid vulnerabilities
- Public disclosure occurs after a fix is available

Thank you for helping keep this project and its users safe!
