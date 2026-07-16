> ## Documentation Index
> Fetch the complete documentation index at: https://notes.kodekloud.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Install Config

> This article explains how package managers automate software deployment and provides instructions for installing Nginx on various platforms.

Before installing Nginx, it’s important to understand how package managers automate software deployment. Package managers handle downloading, dependency resolution, configuration, and removal—streamlining what would otherwise be manual and error-prone tasks.

<Frame>
  ![The image is an infographic titled "Package Manager" showing three functions: downloading software, sorting out dependencies, and managing everything in the system.](https://kodekloud.com/kk-media/image/upload/v1752882303/notes-assets/images/Nginx-For-Beginners-Install-Config/package-manager-infographic-functions.jpg)
</Frame>

## How a Package Manager Works

1. **Retrieve packages and metadata**\
   The manager pulls software from centralized repositories.

<Frame>
  ![The image illustrates the working of a package manager, showing the interaction between software packages and centralized repositories.](https://kodekloud.com/kk-media/image/upload/v1752882304/notes-assets/images/Nginx-For-Beginners-Install-Config/package-manager-repositories-interaction.jpg)
</Frame>

2. **Maintain repositories**\
   OS vendors, third-party developers, and communities (e.g., Nginx Open Source Core) keep these repositories up to date.

<Frame>
  ![The image illustrates the working of a package manager, showing centralized repositories connected to operating system creators and third-party developers.](https://kodekloud.com/kk-media/image/upload/v1752882305/notes-assets/images/Nginx-For-Beginners-Install-Config/package-manager-repositories-illustration.jpg)
</Frame>

3. **Resolve dependencies**\
   Missing libraries or tools are fetched automatically to ensure smooth installs.

<Frame>
  ![The image illustrates the working of a package manager, showing it fetching missing dependencies to ensure smooth operation.](https://kodekloud.com/kk-media/image/upload/v1752882305/notes-assets/images/Nginx-For-Beginners-Install-Config/package-manager-fetching-dependencies.jpg)
</Frame>

4. **Assist with configuration**\
   During or after installation, many managers can apply default or custom configurations.

<Frame>
  ![The image illustrates the process of a package manager, showing three stages: fetching the package, resolving dependencies, and an unspecified final step represented by a gear icon.](https://kodekloud.com/kk-media/image/upload/v1752882307/notes-assets/images/Nginx-For-Beginners-Install-Config/package-manager-process-diagram.jpg)
</Frame>

<Callout icon="lightbulb" color="#1CB2FE">
  Using a package manager reduces manual steps, prevents version conflicts, and keeps your system secure by applying updates consistently.
</Callout>

***

## Popular Package Managers

| Package Manager | Platform               | Repository Type | Official Docs                                    |
| --------------- | ---------------------- | --------------- | ------------------------------------------------ |
| APT             | Debian / Ubuntu        | `.deb`          | [APT Documentation](https://wiki.debian.org/Apt) |
| YUM             | RHEL / CentOS / Fedora | `.rpm`          | [YUM Documentation](https://yum.baseurl.org/)    |
| Homebrew        | macOS / Linux          | Formula         | [Homebrew](https://brew.sh/)                     |
| Chocolatey      | Windows                | NuGet packages  | [Chocolatey](https://chocolatey.org/)            |

### APT (Advanced Package Tool)

APT is the default package manager on Debian-based distributions. It works with `.deb` packages to install, update, and remove software seamlessly.

<Frame>
  ![The image is a slide about the Advanced Package Tool (APT), a package manager used for Debian-based systems like Ubuntu, which works with .deb packages.](https://kodekloud.com/kk-media/image/upload/v1752882307/notes-assets/images/Nginx-For-Beginners-Install-Config/apt-package-manager-debian-ubuntu.jpg)
</Frame>

```bash theme={null}
# Refresh package index
sudo apt update

# Install a package
sudo apt install package_name

# Upgrade all installed packages
sudo apt upgrade

# Remove a package
sudo apt remove package_name

# Search for a package
sudo apt-cache search package_name
```

### YUM (Yellowdog Updater, Modified)

Used by RHEL, CentOS, and Fedora, YUM automates `.rpm` package handling and dependency checks.

```bash theme={null}
# Install a package
sudo yum install package_name

# Remove a package
sudo yum remove package_name

# Update all packages
sudo yum update

# Search for a package
sudo yum search package_name
```

### Homebrew

Homebrew installs software into your home directory on macOS (and Linux), avoiding the need for `sudo` in most cases.

<Frame>
  ![The image is a slide titled "Popular Package Managers" featuring "Homebrew," which is used for macOS and installs software in the home directory without needing sudo privileges.](https://kodekloud.com/kk-media/image/upload/v1752882308/notes-assets/images/Nginx-For-Beginners-Install-Config/popular-package-managers-homebrew.jpg)
</Frame>

```bash theme={null}
# Update Homebrew itself
brew update

# Install a package
brew install package_name

# Uninstall a package
brew uninstall package_name

# Search for a package
brew search package_name
```

### Chocolatey

Chocolatey provides Windows users with a CLI for automated installs via PowerShell.

<Frame>
  ![The image is a slide titled "Popular Package Managers" featuring "Chocolatey," which is used for Windows users to automate software installations from the command line.](https://kodekloud.com/kk-media/image/upload/v1752882309/notes-assets/images/Nginx-For-Beginners-Install-Config/popular-package-managers-chocolatey.jpg)
</Frame>

```powershell theme={null}
# Install a package
choco install package_name

# Uninstall a package
choco uninstall package_name
```

> Always prefer a package manager over manual software installs. It ensures consistent updates, security patches, and simplified maintenance.

***

## Installing Nginx

Follow these commands to install Nginx on your platform. On Linux, append `-y` to bypass confirmation prompts.

```bash theme={null}
# Ubuntu / Debian
sudo apt update
sudo apt install -y nginx

# Red Hat / Fedora
sudo yum update -y
sudo yum install -y nginx

# macOS (Homebrew)
brew update
brew install nginx
```

```powershell theme={null}
# Windows (Chocolatey)
choco install nginx
```

<Callout icon="triangle-alert" color="#FF6B6B">
  Make sure you have the necessary privileges (`sudo` on Unix or an Administrator PowerShell on Windows) before running these commands.
</Callout>

In the next lesson, we will demonstrate each of these installation steps in action, including service management and basic configuration.

<CardGroup>
  <Card title="Watch Video" icon="video" cta="Learn more" href="https://learn.kodekloud.com/user/courses/nginx-for-beginners/module/0de43784-b08d-4ce0-8470-a7541b78fe58/lesson/1b1f4be9-32f7-441a-ae68-f5dd0e0a9c6f" />
</CardGroup>