# Security Hardening Guide - NBA v5.0 BETA

**Last Updated:** 2025-12-18
**Status:** Production-Ready Security Implementation

---

## Overview

This document describes the security hardening measures implemented in NBA v5.0 BETA to protect API keys, prevent unauthorized access, and mitigate API call failures.

---

## Security Features Implemented

### 1. API Key Validation at Startup

**Location:** `src/utils/security.py`

- **Fail-Fast Validation:** System validates all required API keys at startup
- **Missing Keys:** Service fails immediately if required keys are missing (no silent failures)
- **Key Masking:** API keys are masked in logs (e.g., `****1234` instead of full key)
- **Validation Checks:**
  - Required keys must be set and non-empty
  - Keys must be minimum length (10 characters)
  - Database password validation

**Usage:**
```python
from src.utils.security import fail_fast_on_missing_keys

# In startup event
fail_fast_on_missing_keys()  # Raises SecurityError if keys missing
```

### 2. API Authentication (Optional)

**Location:** `src/utils/api_auth.py`

- **Optional Middleware:** API key authentication can be enabled/disabled
- **Flexible:** Supports header (`X-API-Key`) or query parameter (`api_key`)
- **Exempt Paths:** Health checks, metrics, and docs are always accessible
- **Backward Compatible:** Disabled by default for development

**Enable in Production:**
```env
SERVICE_API_KEY=your_strong_random_key_here
REQUIRE_API_AUTH=true
```

**Usage:**
```python
from src.utils.api_auth import get_api_key

@app.get("/protected")
async def protected_endpoint(api_key: str = Security(get_api_key)):
    # Endpoint requires valid API key
    return {"message": "Access granted"}
```

### 3. Circuit Breaker Pattern

**Location:** `src/utils/circuit_breaker.py`

- **Prevents Cascading Failures:** Stops requests when external APIs are failing
- **Automatic Recovery:** Tests recovery and closes circuit when service recovers
- **Configurable:** Failure threshold, timeout, and success threshold
- **Per-Service:** Separate circuit breakers for each external API

**Features:**
- **CLOSED:** Normal operation, requests allowed
- **OPEN:** Service failing, requests rejected immediately
- **HALF_OPEN:** Testing recovery, limited requests allowed

**Configuration:**
- Failure threshold: 5 failures opens circuit
- Success threshold: 2 successes closes circuit (from half-open)
- Timeout: 60 seconds before attempting recovery

**Usage:**
```python
from src.utils.circuit_breaker import get_odds_api_breaker

breaker = get_odds_api_breaker()
result = await breaker.call_async(fetch_odds_function)
```

### 4. Azure Container Apps Hardening

**Location:** `infra/nba/*.bicep`

**Implemented:**
- ✅ **Secrets Management:** Container App secrets via Azure (no local secrets)
- ✅ **Resource Limits:** CPU/memory set in infra
- ✅ **Ingress Controls:** HTTPS-only via ACA ingress
- ✅ **Health Checks:** `/health` used for readiness
- ✅ **Scaling Guards:** min/max replicas + concurrency configured

### 5. API Key Protection

**Prevention of Logging:**
- API keys are masked in all log output
- Sensitive data is sanitized before logging
- No API keys in error messages or stack traces

**Example:**
```python
from src.utils.security import mask_api_key, sanitize_for_logging

# Mask single key
masked = mask_api_key("abc123xyz789")  # Returns: "****789"

# Sanitize entire dictionary
safe_data = sanitize_for_logging({"apiKey": "secret", "data": "public"})
# Returns: {"apiKey": "****", "data": "public"}
```

### 6. Enhanced Error Handling

**Circuit Breaker Integration:**
- External API calls wrapped in circuit breakers
- Automatic retry with exponential backoff (existing)
- Circuit breaker prevents retry storms when service is down

**Error Messages:**
- No sensitive data in error messages
- Clear, actionable error messages
- Proper HTTP status codes

---

## Configuration

### Environment Variables

**Required:**
- `THE_ODDS_API_KEY` - The Odds API key
- `API_BASKETBALL_KEY` - API-Basketball key
- `DB_PASSWORD` - Database password

**Optional (Security):**
- `SERVICE_API_KEY` - API authentication key
- `REQUIRE_API_AUTH` - Enable API authentication (`true`/`false`)
- `ALLOWED_ORIGINS` - CORS allowed origins

### Startup Validation

The system validates all required configuration at startup:

```python
# In src/serving/app.py startup event
fail_fast_on_missing_keys()  # Validates API keys
```

**Validation Results:**
- ✅ **Valid:** Service starts normally
- ❌ **Invalid:** Service fails immediately with clear error message

---

## Production Deployment Checklist

### Before Production:

- [ ] Set strong `DB_PASSWORD` (minimum 12 characters)
- [ ] Set `SERVICE_API_KEY` (generate with `openssl rand -hex 32`)
- [ ] Set `REQUIRE_API_AUTH=true` to enable API authentication
- [ ] Configure `ALLOWED_ORIGINS` to your actual frontend domain(s)
- [ ] Review resource limits in `infra/nba/main.bicep`
- [ ] Ensure HTTPS/TLS via ACA ingress
- [ ] Use Azure Container App secrets / Key Vault
- [ ] Enable monitoring and alerting
- [ ] Review and test circuit breaker thresholds

### Security Best Practices:

1. **API Keys:**
   - Use strong, random keys
   - Rotate keys periodically
   - Never commit keys to version control
   - Use different keys for dev/staging/production

2. **Database:**
   - Use strong passwords (minimum 12 characters)
   - Change default passwords
   - Limit database access to application containers only
   - Enable SSL/TLS for database connections (production)

3. **Network:**
   - Use HTTPS/TLS for all external communication
   - Configure firewall rules
   - Limit exposed ports
   - Use VPN or private networks for internal services

4. **Monitoring:**
   - Monitor API key usage
   - Alert on authentication failures
   - Track circuit breaker state changes
   - Monitor resource usage

---

## Testing Security

### Validate API Keys:
```bash
# Check health endpoint (shows key status)
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health"
```

### Test API Authentication:
```bash
# Without API key (should fail if REQUIRE_API_AUTH=true)
curl "https://$FQDN/slate/today"

# With API key
curl -H "X-API-Key: your_service_api_key" "https://$FQDN/slate/today"
```

### Test Circuit Breaker:
```bash
# Simulate API failures - circuit breaker should open after 5 failures
# Check circuit breaker stats in logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg | grep circuit
```

---

## Troubleshooting

### Service Fails to Start

**Error:** `SecurityError: Missing required API keys`

**Solution:**
1. Check Container App environment variables / Key Vault secrets
2. Verify keys are not empty
3. Check Container App environment variable configuration

### API Authentication Not Working

**Issue:** Requests rejected with 401/403

**Solution:**
1. Verify `SERVICE_API_KEY` is set
2. Check `REQUIRE_API_AUTH=true` is set
3. Verify API key in request header: `X-API-Key: your_key`
4. Check exempt paths (health, metrics, docs should work)

### Circuit Breaker Stuck Open

**Issue:** Circuit breaker won't close

**Solution:**
1. Check external API is actually working
2. Verify network connectivity
3. Check circuit breaker timeout (default: 60 seconds)
4. Review circuit breaker stats: `breaker.get_stats()`

---

## Additional Security Recommendations

### Future Enhancements:

1. **Rate Limiting per API Key:**
   - Track usage per key
   - Implement per-key rate limits

2. **API Key Rotation:**
   - Support multiple keys during rotation
   - Automatic key rotation

3. **Audit Logging:**
   - Log all API key usage
   - Track authentication attempts
   - Monitor suspicious activity

4. **Secret Management:**
   - Integrate with AWS Secrets Manager
   - Use HashiCorp Vault
   - Azure Container App secrets / Key Vault

5. **Network Security:**
   - Private networks for internal services
   - VPN for remote access
   - Firewall rules

---

## Summary

The NBA v5.0 BETA system now includes:

✅ **Startup validation** - Fails fast if API keys missing
✅ **API authentication** - Optional but recommended for production
✅ **Circuit breakers** - Prevents cascading API failures
✅ **Key masking** - API keys never logged
✅ **Azure hardening** - Resource limits and validation
✅ **Enhanced error handling** - No sensitive data in errors

**Security Status:** **Production-Ready** with recommended authentication enabled

---

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Azure Container Apps security](https://learn.microsoft.com/azure/container-apps/secure)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
