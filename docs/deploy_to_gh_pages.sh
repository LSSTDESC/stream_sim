#!/bin/bash
# Script to build documentation and deploy to gh-pages branch

set -e  # Exit on error

echo "=========================================="
echo "Building and Deploying Documentation"
echo "=========================================="

# Check we're on the MUST_BRANCH branch
MUST_BRANCH="documentation_update"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$MUST_BRANCH" ]; then
    echo "Error: This script must be run from the '$MUST_BRANCH' branch"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

#echo "✓ On $MUST_BRANCH branch with no uncommitted changes"

# Build the documentation
echo ""
echo "Building documentation..."
make clean
make html

echo "✓ Documentation built successfully"

# Save the HTML to a temporary location
TMP_DIR=$(mktemp -d)
cp -r build/html/* "$TMP_DIR/"
echo "✓ HTML files copied to temporary location: $TMP_DIR"

# Go back to project root
cd ..

# Stash any untracked files to prevent them from being carried over to gh-pages
echo ""
echo "Stashing untracked files..."
git stash push --include-untracked --keep-index -m "Temporary stash for gh-pages deployment"
STASHED=$?

# Switch to gh-pages branch
echo ""
echo "Switching to gh-pages branch..."
git checkout gh-pages

# Remove old files (except .git and .nojekyll)
echo "Removing old documentation files..."
find . -maxdepth 1 ! -name '.git' ! -name '.nojekyll' ! -name '.' -exec rm -rf {} \; 2>/dev/null || true

# Copy new files
echo "Copying new documentation..."
cp -r "$TMP_DIR"/* .
touch .nojekyll

# Add and commit
echo ""
echo "Committing changes..."
git add -A
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "Update documentation ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "✓ Changes committed"
    
    # Push to remote
    echo ""
    echo "Pushing to GitHub..."
    git push origin gh-pages
    echo "✓ Documentation deployed successfully!"
fi

# Clean up
rm -rf "$TMP_DIR"

# Switch back to MUST_BRANCH branch
echo ""
echo "Switching back to $MUST_BRANCH branch..."
git checkout $MUST_BRANCH

# Restore stashed untracked files if they were stashed
if [ $STASHED -eq 0 ]; then
    echo ""
    echo "Restoring untracked files..."
    git stash pop
fi

echo ""
echo "=========================================="
echo "✓ Documentation deployment complete!"
echo "=========================================="
echo ""
echo "Your documentation should be available at:"
echo "https://lsstdesc.github.io/stream_sim/"
echo ""
echo "Note: It may take a few minutes for GitHub Pages to update."
