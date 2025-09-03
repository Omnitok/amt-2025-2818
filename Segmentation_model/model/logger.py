import git
import os
from datetime import datetime


class StateLogger:
    def __init__(self, output_path):

        self.output_path = output_path
        self.git_repo = git.Repo(self._get_git_root(__file__))

    @staticmethod
    def _get_git_root(path):

        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root

    def dump(self, args):

        self._dump_diff_status()
        self._dump_train_status(args)

    def _dump_diff_status(self):

        with open(os.path.join(self.output_path, "git_state.txt"), "w") as text_file:

            for item in self.git_repo.index.diff(None, create_patch=True):
                text_file.write(str(item))

            for item in self.git_repo.index.diff("HEAD", create_patch=True):
                text_file.write(str(item.a_path))

    def _dump_train_status(self, args):

        with open(os.path.join(self.output_path, "train_state.txt"), "w") as text_file:
            text_file.write("Arguments:" + "\n")
            text_file.write(str(args) + "\n")

            text_file.write("Git commit:" + "\n")
            text_file.write(str(self.git_repo.head.reference.log()) + "\n")
