# GroupProject

A shared codebase for analyzing flight data with PySpark and JupyterLab.

---

## Prerequisites

- **Python 3.8+**  
  (macOS: `brew install python3`; Windows: install from python.org)
- **Java 11** (OpenJDK)  
- **Apache Spark** (optional; we’ll pull it in via `pyspark`)  
- **Git**  
- **GitHub CLI** (`gh`)  
- **JupyterLab**

> 💡 _Optional_: create and activate a virtual environment to isolate dependencies  
> ```sh
> python3 -m venv .venv
> source .venv/bin/activate   # macOS/Linux
> .venv\Scripts\activate      # Windows PowerShell
> ```

---

## Installation

### macOS

```sh
# 1) Update Homebrew & install core tools
brew update
brew install python3 jupyterlab git gh openjdk@11

# 2) Configure Java for Spark
cat << 'EOF' >> ~/.bash_profile
# OpenJDK 11 for Spark
sudo ln -sfn "$(brew --prefix openjdk@11)"/libexec/openjdk.jdk \
             /Library/Java/JavaVirtualMachines/openjdk-11.jdk
export JAVA_HOME=$(/usr/libexec/java_home -v11)
export PATH="$JAVA_HOME/bin:$PATH"
EOF

# 3) Reload your shell
source ~/.bash_profile

# 4) Verify
java -version
```

### Windows

```powershell
# 1) Install Python 3.8+ if needed (from python.org)
# 2) In PowerShell or CMD:
pip install jupyterlab               # pip comes with Python 3.x
winget install --id Git.Git -e --source winget
winget install --id GitHub.cli
```

---

## Clone & Setup

1. **Authenticate** with GitHub CLI for your organization repo:
   ```sh
   gh auth login
   ```
   Follow the prompts. You should see:
   ```
   ✓ Authentication complete.
   ✓ gh config set -h github.com git_protocol https
   ✓ Configured git protocol
   ✓ Logged in as <YourUserName>
   ```
2. **Clone** the repo:
   ```sh
   git clone https://github.com/SmoothData-BigBrain/GroupProject.git
   cd GroupProject
   ```
3. **Create the data folder** and place your CSVs:
   ```sh
   mkdir -p data
   # Copy your extracted airlines files into data folder
   ```
   Your directory should look like:
   ```
   GroupProject/
   ├── data/
   │   └── archive/
   │       └── raw/
   │           ├── file1.csv
   │           └── file2.csv
   │       ├── file1.csv
   │       └── file2.csv
   ├── notebooks/
   ├── requirements.txt
   └── README.md
   ```

---

## Usage

1. **Install Python dependencies**  
   ```sh
   pip install -r requirements.txt
   ```
   *(includes `pyspark`, `seaborn`, etc.)*

2. **Launch JupyterLab**  
   ```sh
   jupyter lab .
   ```
   A browser window will open to your notebook tree.

3. **Open** the notebook in `notebooks/` (or write your own) and start analyzing!

---

### Troubleshooting

- If `java -version` still fails, double-check your `JAVA_HOME` and that you re-sourced your shell.  
- On macOS zsh, you may need to update `~/.zshrc` instead of `~/.bash_profile`.  
- If CSV schema inference is slow, define a schema in your Spark code or set `samplingRatio` (see notebook comments).

---

Happy coding, Jedi Knight!
