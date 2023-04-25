# Data Science Project Organization


## General Organization
- On the cluster, you have your own home directory where you can store files
- There is also the possibility to create temporary workspaces where you can store much larger volumes of data

- We will use a shared workspace to store all the data and the conda environment
- The codebase will live in our individual home directories and be managed with git
- This way, everybody can work on their own code in their directories and collaboration will be ensured with git. But all the data will live in a shared workspace

---

### Shared Workspace: 
- I created shared workspace and granted all of you aread & write access
- Workspace can be found under
```bash
/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/
```
---

### GitHub Repository:
- I created a repository and added all of you as collaborators
- Clone this in your $home directory (`cd ~`) using
```bash
git clone https://github.com/MGenschow/DS_Project.git
```
- The caveat of cloning via HTTPS is that Github will ask for user name and PAT (personal access token) whenever communicating with the remote repository. Using SSH Authentication (i.e. SSH keys) one gets rid of this problem. The following instructions cover how to create SSH keys in the home directory on the cluster and deposit them in Github.
    - Generate SSH key on Cluster. Make sure that you are in your home directory (`cd ~`) and make sure you use the emai address that is associated with your Github account.
    ```bash
    ssh-keygen -t ed25519 -C "stefan.glaisner@student.uni-tuebingen.com"
    ```
    - You will then be asked in which file you want to save the key. But you can simply press Enter to accept the default file location.
    - You will then also be asked to enter a passphrase but this can also be neglected by pressing Enter (twice).
    - Start the the ssh-agent.
    ```bash
    eval "$(ssh-agent -s)"
    ```
    - Add your SSH private key to the ssh-agent.
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```
    - Print the contents of the id_ed25519.pub file.
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    - Copy the content to your clipboard.
    - Go to Github and click on your profile picture.
    - Navigate to Settings > Access > SSH and GPG keys.
    - Click "New SSH key".
    - Choose a title of your choice.
    - Insert the contents of your clipboard into the "Key" pane and click "Add SSH key".
    - After your first `git pull` command you will be asked if you are sure you want to continue connecting. Simply type "yes".
    - Now you should be all set and not be asked for your Github credentials any more.

- For further information, click through this [Github page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and/or this [YouTube video](https://www.youtube.com/watch?v=WgZIv5HI44o).

---

### Conda Environment
- I installed the latest conda version in the workspace in folder `/conda`
- To use conda, you need to first source it using
```bash
    source /pfs/work7/workspace/scratch/tu_zxmav84-ds_project/conda/bin/activate
```
- Alternatively, add this path to your .bashrc/.zshrc file to automatically source when logging into the cluster
- After conda is sourced, you can activate the environment for our project using 
```bash
    conda activate ds_project
```
- Keep in mind to only use conda when installing packages (e.g. `conda install pandas`) and not pip

---

### Using JupyterHub on the cluster:
It is in general possible to use JupyterHub easily on the cluster without going through terminal and SSH. However, using our shared conda environment is not striaghforward here. I advise to rather use code-server (see below)
- Go to this link: [https://uc2-jupyter.scc.kit.edu/](https://uc2-jupyter.scc.kit.edu/)
- Log in via Uni Tuebingen, verify OTP
- Select node resources
- JupyterLab starts as soon as requested resources are availabel
- To log out: Select *File* > *LogOut*

--- 

### Local VSCode Installation:
In general, you can use your local VSCode installation and connect the cluster via SSH. 

But: 

- Can only connect to Login Nodes
- Cannot request resources
- Not advised!

--- 
 
### Using VS Code using Code-Server

- [https://www.nhr.kit.edu/userdocs/horeka/debugging_codeserver/](https://www.nhr.kit.edu/userdocs/horeka/debugging_codeserver/)
- Code-server allows to run VSCode server on any machine
- First you need to allocate resources you want to use, then start code server
- If code server is running, connect local computer to remote server via a SSH tunnel

**Code Server Workflow**
1. Create SSH Connection
    
    ```bash
    ssh tu_zxmav84@bwunicluster.scc.kit.edu
    ```
    
2. Allocate Resources

    Adapt this to the needs that you have.  

See here for  documentation: https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm
    
    ```bash
    salloc -p dev_single -t 30:00 --mem=5000
    ```
    
3. Load code-server module and start code-server

    port can be chosen freely in the unprivileged range above 1024

    You may need to adapt the port if 8081 is already in use by another user
    
    ```bash
    module load devel/code-server
    PASSWORD=test code-server --bind-addr 0.0.0.0:8081 --auth password
    ```
    
4. Create SSH tunnel to the compute node and the port you just created
    
    ```bash
    ssh -L 8081:<NodeID>:8081 tu_zxmav84@bwunicluster.scc.kit.edu
    ```
    
5. Go to your browser and access code-server at (maybe adjust to another port)
    
    [http://127.0.0.1:8081/](http://127.0.0.1:8081/)
    
    
Steps 2 & 3 can alternatively be done using a so-called slurm batch script (see for example [organization/server.sh](https://github.com/MGenschow/DS_Project/blob/main/organization/server.sh)). Here the requested resources can similarly be specified (replaces so to speak the `salloc` command). In addition, the commands `module load devel/code-server`  as well as `PASSWORD=test code-server --bind-addr 0.0.0.0:8081 --auth password` have to be integrated into the batch script as well to finally start code-server. The major advantage of this approach is that the terminal from which code-server has been initialized does not have to remain open for the entire job but can be closed after successful batch submission.

When the requested resources have been allocated (check by `squeue`), one can continue with step 5 & 6.

---

### Things you need to do once on code-server:
#### Define Kernel
1. Install Python Intellisense extension
2. Open new notebook and click on "Select Kernel"
3. The conda environment `ds_project` (see above) should be available to select

#### Create Multi-root Workspace
- You eventually want to see both your $home environment AND the shared workspace in VS Code file browser
- See here for documentation on Multi-Root Workspaces: https://code.visualstudio.com/docs/editor/multi-root-workspaces

- Easiest way I found: 
    - Open your home directory in the file browser
    - Click "Add Folder to Workspace" and add our shared workspace
    - Click "Save Workspace As..." and save it so you can use it in your nect session

    



