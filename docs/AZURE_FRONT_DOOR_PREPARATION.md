# Azure Front Door Preparation

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Internet ────► Container App (nba-gbsv-api)                       │
│                  └── Direct HTTPS ingress                           │
│                  └── CORS configured                                │
│                  └── No WAF                                         │
│                  └── No CDN                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Target Architecture (With Front Door)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TARGET ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Internet ────► Azure Front Door (Global Entry Point)              │
│                  ├── Custom Domain: api.greenbiersportventures.com  │
│                  ├── WAF Policy (DDoS, Rate Limiting)               │
│                  ├── SSL Termination                                │
│                  ├── CDN Caching (static responses)                 │
│                  └── Health Probes                                  │
│                          │                                          │
│                          ▼                                          │
│                  Origin Group                                        │
│                  ├── Primary: Container App (East US)               │
│                  └── Failover: Container App (West US) [future]     │
│                          │                                          │
│                          ▼                                          │
│                  Container App (nba-gbsv-api)                       │
│                  └── Internal ingress only (via Front Door)        │
│                  └── Private networking                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Readiness Assessment

### ✅ What's Already Ready

| Component | Status | Notes |
|-----------|--------|-------|
| Container App | ✅ Ready | Exposed via HTTPS ingress |
| Health endpoint | ✅ Ready | `/health` configured |
| CORS policy | ✅ Ready | Configured in `containerApp.bicep` |
| Liveness/Readiness probes | ✅ Ready | HTTP probes on `/health` |
| Log Analytics | ✅ Ready | Centralized logging |
| App Insights | ✅ Ready | APM configured |

### ⚠️ What Needs Updates for Front Door

| Component | Current | Needed |
|-----------|---------|--------|
| Container App ingress | External | Internal (via Front Door) |
| Custom domain | Not configured | `api.greenbiersportventures.com` |
| WAF Policy | None | Standard protection rules |
| Origin validation | None | X-Azure-FDID header check |
| Rate limiting | None | Per-IP throttling |

---

## Infrastructure Changes Required

### 1. Add Front Door Resource (main.bicep)

```bicep
// Add to main.bicep after Container App

// =============================================================================
// NETWORKING LAYER (Front Door)
// =============================================================================

@description('Front Door profile name')
param frontDoorProfileName string = 'gbs-frontdoor'

@description('Front Door endpoint name')
param frontDoorEndpointName string = 'nba-api'

@description('Custom domain for API')
param customDomain string = 'api.greenbiersportventures.com'

@description('Enable Front Door')
param enableFrontDoor bool = true

// Front Door Profile (Standard/Premium tier for WAF)
resource frontDoorProfile 'Microsoft.Cdn/profiles@2023-05-01' = if (enableFrontDoor) {
  name: frontDoorProfileName
  location: 'global'
  tags: tags
  sku: {
    name: 'Standard_AzureFrontDoor'  // Use 'Premium_AzureFrontDoor' for WAF
  }
}

// Front Door Endpoint
resource frontDoorEndpoint 'Microsoft.Cdn/profiles/afdEndpoints@2023-05-01' = if (enableFrontDoor) {
  parent: frontDoorProfile
  name: frontDoorEndpointName
  location: 'global'
  properties: {
    enabledState: 'Enabled'
  }
}

// Origin Group
resource originGroup 'Microsoft.Cdn/profiles/originGroups@2023-05-01' = if (enableFrontDoor) {
  parent: frontDoorProfile
  name: 'nba-api-origin-group'
  properties: {
    loadBalancingSettings: {
      sampleSize: 4
      successfulSamplesRequired: 3
      additionalLatencyInMilliseconds: 50
    }
    healthProbeSettings: {
      probePath: '/health'
      probeRequestType: 'GET'
      probeProtocol: 'Https'
      probeIntervalInSeconds: 30
    }
    sessionAffinityState: 'Disabled'
  }
}

// Origin (Container App)
resource origin 'Microsoft.Cdn/profiles/originGroups/origins@2023-05-01' = if (enableFrontDoor) {
  parent: originGroup
  name: 'container-app-origin'
  properties: {
    hostName: containerApp.outputs.containerAppFqdn
    httpPort: 80
    httpsPort: 443
    originHostHeader: containerApp.outputs.containerAppFqdn
    priority: 1
    weight: 1000
    enabledState: 'Enabled'
  }
}

// Route
resource route 'Microsoft.Cdn/profiles/afdEndpoints/routes@2023-05-01' = if (enableFrontDoor) {
  parent: frontDoorEndpoint
  name: 'default-route'
  properties: {
    customDomains: []  // Add custom domain reference here
    originGroup: {
      id: originGroup.id
    }
    originPath: '/'
    ruleSets: []
    supportedProtocols: ['Https']
    patternsToMatch: ['/*']
    forwardingProtocol: 'HttpsOnly'
    linkToDefaultDomain: 'Enabled'
    httpsRedirect: 'Enabled'
    enabledState: 'Enabled'
  }
}

// Output Front Door URL
output frontDoorUrl string = enableFrontDoor ? 'https://${frontDoorEndpoint.properties.hostName}' : ''
output frontDoorId string = enableFrontDoor ? frontDoorProfile.properties.frontDoorId : ''
```

### 2. Update Container App for Internal Ingress

```bicep
// In modules/containerApp.bicep - add parameter
@description('Whether ingress is internal only (for Front Door)')
param internalIngress bool = false

// Update ingress configuration
configuration: {
  ingress: {
    external: !internalIngress  // false when behind Front Door
    targetPort: targetPort
    transport: transport
    // Add Front Door ID validation (when internal)
    customDomains: []
  }
  // ...
}
```

### 3. Add Origin Validation Middleware

```python
# src/middleware/frontdoor.py

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os

class FrontDoorValidationMiddleware(BaseHTTPMiddleware):
    """Validate requests come from Azure Front Door."""
    
    def __init__(self, app, front_door_id: str = None):
        super().__init__(app)
        self.front_door_id = front_door_id or os.getenv("AZURE_FRONT_DOOR_ID")
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation if not configured (local dev)
        if not self.front_door_id:
            return await call_next(request)
        
        # Validate X-Azure-FDID header
        fd_id = request.headers.get("X-Azure-FDID")
        
        if fd_id != self.front_door_id:
            # Log potential bypass attempt
            raise HTTPException(
                status_code=403,
                detail="Direct access not allowed. Use Front Door."
            )
        
        return await call_next(request)
```

---

## Migration Path

### Phase 1: Prepare (No Downtime)

```powershell
# 1. Add Front Door resources to Bicep (enableFrontDoor=false initially)
# 2. Deploy infrastructure update
pwsh ./infra/nba/deploy.ps1 -Tag $VERSION

# 3. Test Container App still works directly
curl https://nba-gbsv-api.*.azurecontainerapps.io/health
```

### Phase 2: Deploy Front Door (No Downtime)

```powershell
# 1. Enable Front Door
az deployment group create -g nba-gbsv-model-rg -f infra/nba/main.bicep \
  -p enableFrontDoor=true theOddsApiKey=$KEY apiBasketballKey=$KEY imageTag=$VERSION

# 2. Test Front Door endpoint
curl https://nba-api-*.azureedge.net/health

# 3. Both endpoints work simultaneously
```

### Phase 3: DNS Cutover

```powershell
# 1. Add CNAME record
# api.greenbiersportventures.com -> nba-api-*.azureedge.net

# 2. Add custom domain to Front Door
# (via Azure Portal or Bicep update)

# 3. Test custom domain
curl https://api.greenbiersportventures.com/health
```

### Phase 4: Lock Down Container App

```powershell
# 1. Update Container App to internal ingress
# internalIngress=true in Bicep

# 2. Add Front Door ID validation middleware

# 3. Deploy
pwsh ./infra/nba/deploy.ps1 -Tag $VERSION

# 4. Verify direct access is blocked
curl https://nba-gbsv-api.*.azurecontainerapps.io/health  # Should fail
curl https://api.greenbiersportventures.com/health       # Should work
```

---

## File Structure After Front Door

```
infra/
├── modules/
│   ├── containerApp.bicep     # Updated: internalIngress parameter
│   ├── storage.bicep          # No change
│   └── frontDoor.bicep        # NEW: Reusable Front Door module
├── nba/
│   ├── main.bicep             # Updated: Front Door resources
│   ├── deploy.ps1             # No change
│   └── main.json              # Auto-generated
└── README.md

src/
├── middleware/
│   └── frontdoor.py           # NEW: Origin validation
└── api/
    └── main.py                # Updated: Add middleware
```

---

## Environment Variables to Add

| Variable | Value | Purpose |
|----------|-------|---------|
| `AZURE_FRONT_DOOR_ID` | From deployment output | Origin validation |
| `FRONT_DOOR_ENABLED` | `true` | Feature flag |

---

## Estimated Costs

| Resource | SKU | Monthly Cost |
|----------|-----|-------------|
| Front Door Standard | Standard_AzureFrontDoor | ~$35/month + usage |
| Front Door Premium (with WAF) | Premium_AzureFrontDoor | ~$330/month + usage |
| Data transfer (outbound) | Per GB | ~$0.087/GB |

**Recommendation:** Start with Standard tier, upgrade to Premium if WAF needed.

---

## Checklist: Front Door Readiness

### Current State (Ready)
- [x] Container App with HTTP health endpoint
- [x] CORS policy configured
- [x] Liveness/Readiness probes
- [x] Log Analytics integration
- [x] App Insights APM
- [x] Modular Bicep structure

### Pending (Before Front Door)
- [ ] Create `frontDoor.bicep` module
- [ ] Add Front Door parameters to `main.bicep`
- [ ] Create origin validation middleware
- [ ] Configure custom domain DNS
- [ ] Add `internalIngress` parameter to containerApp.bicep
- [ ] Test Front Door health probes against `/health`

### Post-Deployment
- [ ] Verify Front Door endpoint accessible
- [ ] Verify custom domain works
- [ ] Verify direct Container App access blocked
- [ ] Monitor latency via App Insights
- [ ] Configure alerts for origin health
