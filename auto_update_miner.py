import subprocess
import os
import sys
import logging
import importlib
from typing import Optional

# Define repository and script paths
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_PATH = "neurons/miner.py"
BRANCH = 'deap'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_git_command(command: list[str]) -> Optional[str]:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(
            ['git', '-C', REPO_PATH] + command,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        return None

def get_local_version() -> Optional[str]:
    """Read the local version from chain/__init__.py."""
    try:
        sys.path.insert(0, REPO_PATH)
        import chain
        importlib.reload(chain)  # Ensure we're getting the latest version
        return getattr(chain, '__spec_version__', None)
    except ImportError:
        logging.error("Failed to import chain module")
        return None
    finally:
        sys.path.pop(0)

def get_remote_version() -> Optional[str]:
    """Fetch the remote version from chain/__init__.py."""
    if run_git_command(['fetch']) is None:
        return None
    
    try:
        remote = f"origin/{BRANCH}:chain/__init__.py"
        print(remote)
        remote_content = run_git_command(['show', remote])
        if remote_content is None:
            return None
        
        # Extract __spec_version__ from the remote content
        for line in remote_content.split('\n'):
            if line.startswith('__spec_version__'):
                return line.split('=')[1].strip().strip("'\"")
        
        logging.error("__spec_version__ not found in remote chain/__init__.py")
        return None
    except Exception as e:
        logging.error(f"Error fetching remote version: {e}")
        return None

def get_current_branch() -> Optional[str]:
    """Returns the current local git branch."""
    return run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])

def stash_changes() -> bool:
    """Stash local changes to avoid conflicts."""
    return run_git_command(['stash']) is not None

def apply_stash() -> bool:
    """Apply stashed changes after update."""
    return run_git_command(['stash', 'pop']) is not None

def switch_to_branch(branch_name: str) -> bool:
    """Switch to the specified branch."""
    return run_git_command(['checkout', branch_name]) is not None

def update_repo() -> bool:
    """Pull the latest changes from the repository."""
    return run_git_command(['pull']) is not None

def run_main_script():
    """Run the main script."""
    try:
        subprocess.run([sys.executable, MAIN_SCRIPT_PATH], check=True)
        logging.info("Main script executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing main script: {e}")
        return False
    return True

def main():
    try:
        local_version = get_local_version()
        remote_version = get_remote_version()
        current_branch = get_current_branch()

        if not all([local_version, remote_version, current_branch]):
            logging.error("Failed to retrieve necessary information.")
            return 1

        logging.info(f"Local version: {local_version}")
        logging.info(f"Remote version: {remote_version}")

        update_performed = False

        if int(local_version) != int(remote_version):
            logging.info(f"New version detected on the {BRANCH} branch.")
            breakpoint
            if current_branch != BRANCH:
                logging.info(f"Currently on branch {current_branch}.")
                
                status = run_git_command(['status', '--porcelain'])
                
                if status:
                    logging.info("Uncommitted changes found, stashing them.")
                    if not stash_changes():
                        raise Exception("Failed to stash changes.")

                if not switch_to_branch(BRANCH):
                    raise Exception("Failed to switch to main branch.")
                
                if not update_repo():
                    raise Exception("Failed to update repository.")
                
                if not switch_to_branch(current_branch):
                    raise Exception(f"Failed to switch back to {current_branch} branch.")
                
                if status and not apply_stash():
                    raise Exception("Failed to apply stashed changes.")
            else:
                if not update_repo():
                    raise Exception("Failed to update repository.")
            
            logging.info("Update completed successfully.")
            update_performed = True
        else:
            logging.info("No update required.")

        # Run the main script regardless of whether an update was performed
        if not run_main_script():
            return 1

    except Exception as e:
        logging.error(f"An error occurred during the update process: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())