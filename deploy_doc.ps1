# Set error action to stop the script on any error
$ErrorActionPreference = "Stop"

# Define variables
$repoRoot = Get-Location
$docsSourceDir = Join-Path $repoRoot "docs" # The directory containing make.bat and your Sphinx source
$htmlBuildOutputSubDir = "_build/html" # Relative to docsSourceDir, where make.bat puts HTML
$htmlSourceDir = Join-Path $docsSourceDir $htmlBuildOutputSubDir
$tempDocsDir = Join-Path $repoRoot "..\_tmp_docs" # Temporary folder outside the repo
# On gh-pages branch, we typically want the built HTML to be at the root,
# not inside a 'docs' folder, unless your GitHub Pages is configured otherwise.
# For standard gh-pages, the root of the branch serves the content.
# If your GitHub Pages is set to serve from '/docs' then keep $ghPagesTargetDocsDir = Join-Path $repoRoot "docs"
# Otherwise, for root serving, it should be just $repoRoot
$ghPagesTargetDocsDir = $repoRoot # Assuming GitHub Pages serves from the root of the gh-pages branch
$ghPagesBranch = "gh-pages"
$commitMessage = "Update documentation for GitHub Pages"

Write-Host "--- Starting Local GitHub Pages Deployment ---"

# 1. Build the docs using make.bat
Write-Host "1. Building documentation using make.bat..."
# Change directory to where your make.bat is located (inside the docs folder)
Set-Location $docsSourceDir
# Execute the make.bat file to build HTML
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

# 3. Handle uncommitted changes before checking out gh-pages
Write-Host "3. Stashing any local changes before checking out '$ghPagesBranch' branch..."
try {
    # Check if there are any uncommitted changes
    $status = git status --porcelain
    if ($status) {
        Write-Host "Local changes detected. Stashing them..."
        git stash save "Pre-gh-pages-deployment-stash"
    } else {
        Write-Host "No local changes to stash."
    }
} catch {
    Write-Host "Warning: Could not stash changes. This might indicate an issue with your git setup or an already clean working directory. Error: $($_.Exception.Message)"
    # Don't stop here, try to proceed if possible, but the next checkout might fail.
}

# 4. Checkout the gh-pages branch
Write-Host "4. Checking out '$ghPagesBranch' branch..."
try {
    # Ensure all remote branches are fetched
    git fetch origin

    # Attempt to checkout the gh-pages branch
    git checkout $ghPagesBranch
    Write-Host "Checked out existing '$ghPagesBranch' branch."

    # If the branch exists, clean its contents (except .git)
    Write-Host "Cleaning contents of '$ghPagesBranch'..."
    # Get all items in the current directory, excluding the .git folder
    $itemsToRemove = Get-ChildItem -Path $repoRoot -Force | Where-Object { $_.Name -ne ".git" }
    foreach ($item in $itemsToRemove) {
        Remove-Item $item.FullName -Recurse -Force
    }
    Write-Host "Branch contents cleaned."

} catch {
    # If the branch doesn't exist, create it as an orphan branch
    if ($_.Exception.Message -match "did not match any file\(s\) known to git") {
        Write-Host "Branch '$ghPagesBranch' does not exist. Creating as orphan branch."
        git checkout --orphan $ghPagesBranch
        Write-Host "Orphan '$ghPagesBranch' branch created."
        # After creating an orphan branch, the working directory contains all files
        # from the previous branch. We need to remove them.
        git rm -rf .
    } else {
        throw "Failed to checkout or clean '$ghPagesBranch' branch: $($_.Exception.Message)"
    }
}

# 5. Copy content from temporary folder to the gh-pages target directory
Write-Host "5. Copying new documentation from '$tempDocsDir' to '$ghPagesTargetDocsDir'."

# For gh-pages, you typically want the built content to be at the root of the branch.
# So, we copy directly into the $repoRoot which is where we are on the gh-pages branch.
# If you actually need a 'docs' folder on gh-pages, then uncomment the original line:
# New-Item -ItemType Directory -Path $ghPagesTargetDocsDir -ErrorAction SilentlyContinue # Create if it doesn't exist
Copy-Item (Join-Path $tempDocsDir "*") $ghPagesTargetDocsDir -Recurse -Force
Write-Host "New documentation copied."

# 6. Add .nojekyll file
Write-Host "6. Creating .nojekyll file."
Set-Content -Path (Join-Path $repoRoot ".nojekyll") -Value ""
Write-Host ".nojekyll created."

# 7. Stage, commit, and push changes
Write-Host "7. Staging, committing, and pushing changes..."
git add .
try {
    git commit -m $commitMessage
    Write-Host "Changes committed."
} catch {
    Write-Host "No changes to commit (already up to date)."
}

# Push to origin gh-pages, forcing to overwrite history if needed (common for gh-pages)
# Use --force-with-lease for a safer force push if you are in a team environment.
# For personal gh-pages, --force is often acceptable if you are the sole pusher.
Write-Host "Pushing changes to '$ghPagesBranch' branch..."
git push --force origin $ghPagesBranch
Write-Host "Changes pushed to '$ghPagesBranch'."

# 8. Return to original branch (e.g., 'main' or 'master') and pop stash
Write-Host "8. Returning to original branch and restoring stashed changes..."
# Determine the previous branch before switching to gh-pages
# This assumes you were on 'main' or 'master'
# You might want to get the actual previous branch if your workflow is more complex
$originalBranch = "main" # or "master" or whatever your development branch is
git checkout $originalBranch
Write-Host "Returned to '$originalBranch' branch."

# Pop stashed changes, if any
try {
    $stashList = git stash list
    if ($stashList -like "*Pre-gh-pages-deployment-stash*") {
        Write-Host "Restoring stashed changes..."
        git stash pop
        Write-Host "Stashed changes restored."
    } else {
        Write-Host "No stash to apply."
    }
} catch {
    Write-Host "Warning: Could not pop stash. Please check your stash manually. Error: $($_.Exception.Message)"
}

# 9. Clean up temporary documentation directory
Write-Host "9. Cleaning up temporary documentation directory: $tempDocsDir"
try {
    if (Test-Path $tempDocsDir) {
        Remove-Item $tempDocsDir -Recurse -Force
        Write-Host "Temporary documentation directory removed."
    } else {
        Write-Host "Temporary documentation directory not found, nothing to clean."
    }
} catch {
    Write-Host "Warning: Could not remove temporary documentation directory. Please delete it manually if necessary. Error: $($_.Exception.Message)"
}


Write-Host "--- Deployment Complete! ---"