> ## Documentation Index
> Fetch the complete documentation index at: https://notes.kodekloud.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Install Config Introduction

> This guide covers installing and configuring Nginx on various operating systems, including service management and securing your server.

In this guide, you’ll learn how to install and configure Nginx on multiple operating systems. We’ll walk through package managers, Nginx service management, configuration structure, hosting a static website, and securing your server with UFW.

## What You’ll Learn

* Package managers: APT, YUM, Homebrew, and Windows Subsystem for Linux (WSL).
* Installing Nginx on Ubuntu, CentOS, macOS, and Windows (WSL).
* Starting, stopping, reloading, and checking the status of the Nginx service.
* Anatomy of **nginx.conf**: global directives, the HTTP block, and server blocks.
* Hosting a simple static site with Nginx server blocks.
* Configuring ports and securing your site with UFW.

<Frame>
  ![The image lists objectives related to Nginx, including understanding package managers, installing Nginx on various operating systems, managing Nginx services, and exploring configuration settings.](https://kodekloud.com/kk-media/image/upload/v1752882302/notes-assets/images/Nginx-For-Beginners-Install-Config-Introduction/nginx-objectives-installation-management.jpg)
</Frame>

***

We’ll begin by exploring package managers and installing Nginx. Then, we’ll cover service management, dive into the configuration file, set up a static website, and finish with firewall security using UFW.

## 1. Package Manager Overview

Choose the appropriate package manager for your OS:

| Operating System | Package Manager | Install Nginx Command                                     |
| ---------------- | --------------- | --------------------------------------------------------- |
| Ubuntu           | APT             | `sudo apt update && sudo apt install nginx`               |
| CentOS/RHEL      | YUM or DNF      | `sudo yum install epel-release && sudo yum install nginx` |
| macOS            | Homebrew        | `brew update && brew install nginx`                       |
| Windows (WSL)    | APT             | `sudo apt update && sudo apt install nginx`               |

<Callout icon="lightbulb" color="#1CB2FE">
  Ensure you have administrative privileges (`sudo`) before running installation commands.
</Callout>

## 2. Installing Nginx

We’ll cover step-by-step instructions for:

* Ubuntu & Debian
* CentOS & RHEL
* macOS (Homebrew)
* Windows WSL

***

Next, we’ll manage the Nginx service, explore **nginx.conf**, host a static site, and secure your server with UFW.

## Links and References

* [Nginx Official Documentation](https://nginx.org/en/docs/)
* [Ubuntu APT Guide](https://help.ubuntu.com/community/AptGet/Howto)
* [Yum Package Management](https://docs.fedoraproject.org/en-US/quick-docs/dnf/)
* [Homebrew Documentation](https://docs.brew.sh/)
* [UFW Manual](https://help.ubuntu.com/community/UFW)

<CardGroup>
  <Card title="Watch Video" icon="video" cta="Learn more" href="https://learn.kodekloud.com/user/courses/nginx-for-beginners/module/0de43784-b08d-4ce0-8470-a7541b78fe58/lesson/196d68ff-0e61-4b1b-a24b-3ef74ccf275c" />
</CardGroup>