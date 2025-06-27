# Install Build Tools for llama.cpp CUDA Build
# This script will help you install the required tools

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Build Tools for llama.cpp" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  This script may need administrator privileges for some installations." -ForegroundColor Yellow
    Write-Host "If you encounter permission issues, run PowerShell as Administrator." -ForegroundColor Yellow
    Write-Host ""
}

# Check current installations
Write-Host "🔍 Checking current installations..." -ForegroundColor Yellow

$tools = @{
    "Git" = "git"
    "CMake" = "cmake"
    "Visual Studio Build Tools" = "cl"
    "CUDA Toolkit" = "nvcc"
}

$missingTools = @()

foreach ($tool in $tools.GetEnumerator()) {
    try {
        $version = & $tool.Value --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $($tool.Key): Found" -ForegroundColor Green
        } else {
            throw "Not found"
        }
    } catch {
        Write-Host "❌ $($tool.Key): Not found" -ForegroundColor Red
        $missingTools += $tool.Key
    }
}

if ($missingTools.Count -eq 0) {
    Write-Host ""
    Write-Host "🎉 All required tools are already installed!" -ForegroundColor Green
    Write-Host "You can now run: .\build_llama_cuda.bat" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "Missing tools: $($missingTools -join ', ')" -ForegroundColor Red
Write-Host ""

# Installation options
Write-Host "Installation Options:" -ForegroundColor Cyan
Write-Host "1. Open download pages for manual installation" -ForegroundColor White
Write-Host "2. Use winget to install tools (if available)" -ForegroundColor White
Write-Host "3. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Opening download pages..." -ForegroundColor Yellow
        
        $downloads = @{
            "Git" = "https://git-scm.com/download/win"
            "CMake" = "https://cmake.org/download/"
            "Visual Studio Build Tools" = "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
            "CUDA Toolkit" = "https://developer.nvidia.com/cuda-downloads"
        }
        
        foreach ($tool in $missingTools) {
            if ($downloads.ContainsKey($tool)) {
                Write-Host "Opening $tool download page..." -ForegroundColor Green
                Start-Process $downloads[$tool]
                Start-Sleep 2
            }
        }
        
        Write-Host ""
        Write-Host "📋 Installation Instructions:" -ForegroundColor Cyan
        Write-Host "1. Install Git first" -ForegroundColor White
        Write-Host "2. Install CMake (add to PATH during installation)" -ForegroundColor White
        Write-Host "3. Install Visual Studio Build Tools with 'Desktop development with C++'" -ForegroundColor White
        Write-Host "4. Install CUDA Toolkit (optional but recommended)" -ForegroundColor White
        Write-Host ""
        Write-Host "After installation, restart your command prompt and run this script again." -ForegroundColor Yellow
    }
    "2" {
        Write-Host ""
        Write-Host "Attempting to install with winget..." -ForegroundColor Yellow
        
        # Check if winget is available
        try {
            $wingetVersion = winget --version
            Write-Host "✅ winget found: $wingetVersion" -ForegroundColor Green
        } catch {
            Write-Host "❌ winget not available. Please use option 1 for manual installation." -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
        
        # Install tools with winget
        $wingetPackages = @{
            "Git" = "Git.Git"
            "CMake" = "Kitware.CMake"
        }
        
        foreach ($tool in $missingTools) {
            if ($wingetPackages.ContainsKey($tool)) {
                Write-Host "Installing $tool..." -ForegroundColor Green
                winget install $wingetPackages[$tool] --accept-source-agreements --accept-package-agreements
            } else {
                Write-Host "⚠️  $tool not available via winget. Please install manually." -ForegroundColor Yellow
            }
        }
        
        Write-Host ""
        Write-Host "✅ Installation completed!" -ForegroundColor Green
        Write-Host "Please restart your command prompt and run this script again to verify." -ForegroundColor Yellow
    }
    "3" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit 1
    }
}

Read-Host "Press Enter to exit" 