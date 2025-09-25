# Add this script to your powershell startup script to make the scripts and
# tools in this package accessible from any directory on your system.

# Get input arguments
param(
	[string]$ProjectPath,
	[bool]$Verbose
)

$LecroyScriptPath = Join-Path -Path $ProjectPath -ChildPath "apps\tools\view_lecroy.py"
function lecroy {
	param(
		[Parameter(ValueFromRemainingArguments=$true)]
		[string[]]$args
	)
	
	python $LecroyScriptPath @args
}

$ComplecroyScriptPath = Join-Path -Path $ProjectPath -ChildPath "apps\tools\compare_lecroy.py"
function complecroy {
	param(
		[Parameter(ValueFromRemainingArguments=$true)]
		[string[]]$args
	)
	python $ComplecroyScriptPath @args
}

# Success message if requested
if ($Verbose){
	Write-Output "Added ChillyInductor at path:" $ProjectPath
}