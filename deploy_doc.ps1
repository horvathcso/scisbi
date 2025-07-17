# Set error action to stop the script on any error
$ErrorActionPreference = "Stop"

# Define variables
$repoRoot = Get-Location
$docsSourceDir = Join-Path $repoRoot "docs" # The directory containing make.bat and your Sphinx source
$htmlBuildOutputSubDir = "_build/html" # Relative to docsSourceDir, where make.bat puts HTML
$htmlSourceDir = Join-Path $docsSourceDir $htmlBuildOutputSubDir
$tempDocsDir = Join-Path $repoRoot "..\_tmp_docs" # Temporary folder outside the repo
$ghPagesTargetDocsDir = Join-Path $repoRoot "docs" # The target 'docs' folder on gh-pages branch
$ghPagesBranch = "gh-pages"
$commitMessage = "Update documentation for GitHub Pages"

Write-Host "--- Starting Local GitHub Pages Deployment ---"

# 1. Build the docs using make.bat
Write-Host "1. Building documentation using make.bat..."
# Change directory to where your make.bat is located (inside the docs folder)
Set-Location $docsSourceDir
# Execute the make.bat file to build HTML
# The '&' operator is used to run an executable or script in PowerShell
& ".\make.bat" html
# Go back to repository root
Set-Location $repoRoot
Write-Host "Documentation built successfully by make.bat."

# 2. Create a temporary directory and copy built HTML content
Write-Host "2. Copying built docs to temporary directory: $tempDocsDir"
if (Test-Path $tempDocsDir) {
    Remove-Item $tempDocsDir -Recurse -Force
    Write-Host "Removed existing temporary directory."
}
New-Item -ItemType Directory -Path $tempDocsDir
Copy-Item (Join-Path $htmlSourceDir "*") $tempDocsDir -Recurse -Force
Write-Host "Built docs copied to temporary directory."

# 3. Checkout the gh-pages branch
Write-Host "3. Checking out '$ghPagesBranch' branch..."
# Fetch all branches to ensure gh-pages is available locally
git fetch origin
# Try to checkout, if it fails (e.g., branch doesn't exist), create an orphan branch
try {
    git checkout $ghPagesBranch
    # If branch exists, clean its contents (except .git)
    Write-Host "Branch '$ghPagesBranch' exists. Cleaning its contents..."
    # Use git rm -rf . to remove tracked files and Remove-Item for untracked files/dirs
    git rm -rf . | Out-Null # Suppress output if nothing to remove
    Remove-Item (Join-Path $repoRoot "*") -Exclude ".git" -Recurse -Force | Out-Null # Suppress output
} catch {
    Write-Host "Branch '$ghPagesBranch' does not exist. Creating as orphan branch."
    git checkout --orphan $ghPagesBranch
}
Write-Host "Checked out '$ghPagesBranch' branch."

# 4. Remove existing 'docs' folder (if any) and copy content from temporary folder
Write-Host "4. Copying new documentation to the '$ghPagesTargetDocsDir' folder."
if (Test-Path $ghPagesTargetDocsDir) {
    Remove-Item $ghPagesTargetDocsDir -Recurse -Force
    Write-Host "Removed existing '$ghPagesTargetDocsDir' on '$ghPagesBranch'."
}
New-Item -ItemType Directory -Path $ghPagesTargetDocsDir

# Copy contents from the temporary directory into the 'docs' folder
Copy-Item (Join-Path $tempDocsDir "*") $ghPagesTargetDocsDir -Recurse -Force
Write-Host "New documentation copied to '$ghPagesTargetDocsDir'."

# 5. Add .nojekyll file
Write-Host "5. Creating .nojekyll file."
Set-Content -Path (Join-Path $repoRoot ".nojekyll") -Value ""
Write-Host ".nojekyll created."

# 6. Stage, commit, and push changes
Write-Host "6. Staging, committing, and pushing changes..."
git add .
git commit -m $commitMessage

# Push to origin gh-pages, forcing to overwrite history if needed (common for gh-pages)
git push --force origin $ghPagesBranch
Write-Host "Changes pushed to '$ghPagesBranch'."

Write-Host "--- Deployment Complete! ---"
