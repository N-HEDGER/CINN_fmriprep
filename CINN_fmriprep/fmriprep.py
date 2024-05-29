import os
from simple_slurm import Slurm
from .utils import load_pkg_yaml
import pkg_resources
import yaml

#os.system('source ~/.bashrc')
#os.system('cd /storage')

base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("CINN_fmriprep", 'config')))

# Define the path to the config file
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')


activate_command=load_pkg_yaml()['cmds']['activate_cmd']

def activate_fmriprep(line_to_add: str = activate_command):
    """
    Activates fmriprep by adding a line to the ~/.bashrc file.

    Args:
        line_to_add (str, optional): The line to add to the ~/.bashrc file. Defaults to the fmriprep path.
    """
    bashrc_path = os.path.expanduser('~/.bashrc')

    # Check if ~/.bashrc exists
    if not os.path.exists(bashrc_path):
        # Create ~/.bashrc if it doesn't exist
        print('looks like you dont have a ~/.bashrc file, creating one for you...')
        with open(bashrc_path, 'w') as f:
            pass

    # Check if the line is already present in ~/.bashrc
    with open(bashrc_path, 'r') as f:
        lines = f.readlines()
        line_exists = any(line.strip() == line_to_add for line in lines)

    if line_exists:
        print('fmriprep already activated ~/.bashrc')

    # Add the line to ~/.bashrc if it doesn't exist
    if not line_exists:
        with open(bashrc_path, 'a') as f:
            f.write(line_to_add + '\n')
        print('fmriprep activated')

    os.system('source ~/.bashrc')


class FmriPrepHandler:
    """
    A class for handling fMRI preprocessing using fmriprep.

    Args:
        bids_path (str): The path to the BIDS dataset.
        out_path (str): The output path for the preprocessed data.
        workpath (str): The working directory path.
        slurmout_path (str): The path for SLURM output files.
        subject (str, optional): The subject to preprocess. Defaults to 'allsubs'.
        yaml (str, optional): The path to the YAML configuration file. Defaults to pkg_yaml.

    Attributes:
        subject (str): The subject to preprocess.
        jobname (str): The name of the fmriprep job.
        out_path (str): The output path for the preprocessed data.
        workpath (str): The working directory path.
        bids_path (str): The path to the BIDS dataset.
        slurmout_path (str): The path for SLURM output files.
        outfile (str): The path to the SLURM output file.
        errfile (str): The path to the SLURM error file.
        yaml (str): The path to the YAML configuration file.
        y (dict): The loaded YAML configuration.
        tf_path (str): The path to the TensorFlow environment.
        fs_license (str): The path to the FreeSurfer license file.
        slurm (Slurm): The SLURM job object.
        slurm_pre_commands (list): The list of SLURM pre-commands.
        tf_wcard (str): The wildcard for the TensorFlow environment.
        cmd_wcard (str): The wildcard for the fmriprep command.
        cmd (str): The final fmriprep command to be executed.
        message (str): The message about the submitted job.

    Methods:
        make_dirs(): Creates the output and working directories.
        load_yaml(): Loads the YAML configuration file into memory.
        internalize_config(y: dict, subdict: str): Internalizes a sub-dictionary from the YAML configuration.
        make_slurm(additionals: list): Creates the SLURM job script for running fmriprep.
        submit_slurm(): Submits the SLURM job for running fmriprep.
        make_message(): Creates a message with information about the submitted job.
        print_message(): Prints the message about the submitted job.
    """

    def __init__(self, bids_path: str, out_path: str, work_path: str, slurmout_path: str, subject: str = 'allsubs', yaml: str = pkg_yaml):
        self.subject = subject
        self.jobname = 'fmriprep_{subject}'.format(subject=self.subject)
        self.out_path = os.path.join(out_path, self.subject)
        self.work_path = os.path.join(work_path, self.subject)
        self.bids_path = bids_path
        self.make_dirs()

        
        self.subject = subject
        self.slurmout_path = slurmout_path
        self.outfile = os.path.join(self.slurmout_path, '{subject}.out'.format(subject=self.jobname))
        self.errfile = os.path.join(self.slurmout_path, '{subject}.err'.format(subject=self.jobname))

        self.yaml = yaml
        self.load_yaml()
        self.internalize_config(self.y, 'paths')
        self.internalize_config(self.y, 'cmds')
        self.tf_path = self.tf_path
        self.fs_license = self.fs_license

    def make_dirs(self):
        """
        Creates the output and working directories.
        """
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.work_path, exist_ok=True)


    def load_yaml(self):
        """
        Loads the YAML configuration file into memory.
        """
        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

    def internalize_config(self, y: dict, subdict: str):
        """
        Internalizes a sub-dictionary from the YAML configuration.

        Args:
            y (dict): A dictionary containing the YAML configuration.
            subdict (str): The sub-dictionary to internalize.
        """
        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])

    def make_slurm(self, additionals=[], output_spaces=['MNI152NLin2009cAsym']):
        """
        Creates the SLURM job script for running fmriprep.

        Args:
            additionals (list, optional): Additional arguments for fmriprep. Defaults to [].
            output_spaces (str, optional): The output spaces for preprocessing. Defaults to 'MNI152NLin2009cAsym'.
        """
        self.output_spaces = output_spaces
        self.slurm = Slurm(**load_pkg_yaml()['slurm'])

        self.slurm._add_one_argument('output', self.outfile)
        self.slurm._add_one_argument('error', self.errfile)
        [self.slurm.add_cmd(cmd) for cmd in self.slurm_pre_commands]

        self.slurm.add_cmd(self.tf_wcard.format(tf_path=self.tf_path))

        kwarg_string = ' '.join(additionals)
        ospace_string = ' '.join(output_spaces)

        self.cmd = self.cmd_wcard.format(
            outpath=self.out_path,
            subject=self.subject,
            bidspath=self.bids_path,
            fs_license=self.fs_license,
            workpath=self.work_path,
            output_spaces=ospace_string)

        self.cmd = self.cmd + ' ' + kwarg_string
        self.cmd = self.cmd.replace('--participant-label allsubs', '')

    def submit_slurm(self):
        """
        Submits the SLURM job for running fmriprep.
        """
        self.slurm.sbatch(self.cmd)
        self.make_message()
        self.print_message()

    def make_message(self):
        """
        Creates a message with information about the submitted job.
        """
        self.message = ("""fMRIprep Job with {jobname} submitted! \n
               Outputs will be found at {outpath} \n 
              Progress updated in {outfile}\n
              Errors will be reported at {errfile}\n""".format(jobname=self.jobname, outpath=self.out_path, outfile=self.outfile, errfile=self.errfile))

    def print_message(self):
        """
        Prints the message about the submitted job.
        """
        print(self.message)



class MultipleFmriPrepHandler:
    """
    A class that handles multiple FmriPrepHandler instances for each subject.

    Args:
        bids_path (str): The path to the BIDS dataset.
        out_path (str): The output path for the preprocessed data.
        work_path (str): The working directory path.
        slurmout_path (str): The path to the SLURM output directory.
        subjects (list, optional): A list of subject IDs. Defaults to ['1', '2'].

    Attributes:
        subjects (list): A list of subject IDs.
        bids_path (str): The path to the BIDS dataset.
        out_path (str): The output path for the preprocessed data.
        slurmout_path (str): The path to the SLURM output directory.
        workpath (str): The working directory path.
        handlers (list): A list of FmriPrepHandler instances.

    Methods:
        make_fmriprep_handlers: Creates FmriPrepHandler instances for each subject.
        make_slurms: Generates SLURM scripts for each FmriPrepHandler instance.
        submit_slurms: Submits the generated SLURM scripts for execution.
    """

    def __init__(self, bids_path: str, out_path: str, work_path: str, slurmout_path: str, subjects: list = ['1', '2']):
        self.subjects = subjects
        self.bids_path = bids_path
        self.out_path = out_path
        self.slurmout_path = slurmout_path
        self.work_path = work_path
        self.handlers = []

    def make_fmriprep_handlers(self):
        """
        Creates FmriPrepHandler instances for each subject.
        """
        for sub in self.subjects:
            self.handlers.append(FmriPrepHandler(self.bids_path, self.out_path, self.work_path, self.slurmout_path, sub))

    def make_slurms(self, additionals=[], output_spaces=['MNI152NLin2009cAsym']):
        """
        Generates SLURM scripts for each FmriPrepHandler instance.

        Args:
            additionals (list, optional): Additional arguments for SLURM scripts. Defaults to [].
            output_spaces (list, optional): A list of output spaces. Defaults to ['MNI152NLin2009cAsym'].
        """
        for handler in self.handlers:
            handler.make_slurm(additionals, output_spaces)

    def submit_slurms(self):
        """
        Submits the generated SLURM scripts for execution.
        """
        for handler in self.handlers:
            handler.submit_slurm()
