# Define infrastructure variables to exclude from --env-vars
$infraVars = @(
    'APP_NAME',
    'RESOURCE_GROUP',
    'IMAGE_NAME',
    'IMAGE_TAG',
    'ENVIRONMENT_NAME',
    'REGISTRY_SERVER',
    'REGISTRY_USERNAME',
    'REGISTRY_PASSWORD',
    'LOGS_WORKSPACE_ID',
    'LOGS_WORKSPACE_KEY'
)

# Read .env file and set environment variables, collect app vars
$envFile = ".azure/agents/.env"
$envVarsList = @()

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "Env:\$key" -Value $value
            if ($infraVars -notcontains $key) {
                $envVarsList += "$key=$value"
            }
        }
    }
} else {
    Write-Host "Error: .env file not found in current directory"
    exit 1
}

# Join environment variables for --env-vars
$envVarsArg = $envVarsList -join ' '

# Compose the full command as a string
$azCmd = @"
az containerapp up --name $env:APP_NAME --resource-group $env:RESOURCE_GROUP --image $env:REGISTRY_SERVER/multi-agents-agent:$env:IMAGE_TAG --environment $env:ENVIRONMENT_NAME --logs-workspace-id $env:LOGS_WORKSPACE_ID --logs-workspace-key $env:LOGS_WORKSPACE_KEY --registry-server $env:REGISTRY_SERVER --registry-username $env:REGISTRY_USERNAME --registry-password $env:REGISTRY_PASSWORD --ingress external --target-port 8001 --env-vars $envVarsArg
"@

# Print the command
Write-Host "Running the following command:"
Write-Host $azCmd

# Execute the command
Invoke-Expression $azCmd
