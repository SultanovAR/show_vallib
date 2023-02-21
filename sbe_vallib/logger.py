import os
import time
import pickle
import shutil
import pandas as pd


class Logger(object):
    def __init__(
        self,
        path=".",
        level=[
            "valid_warnings",
            "valid_errors",
            "test_warnings",
            "test_errors",
            "semaphore",
            "model_errors",
            "model_warning",
            "progress",
            "short_res",
            "pictures",
            "answers",
            "tests_logs",
            "info",
        ],
        **params
    ):
        """
        Объект, который инициализирует запись логов. При создании нового Logger, имеющийся файл дополняется логами.

        @path: путь до файла, куда будет записываться лог текущего теста и/или валидации
        @level: перечесление элементов, подлежащих логированию:
        """
        if not os.path.exists(path):
            os.mkdir(path)
        self.base_path = path
        self.level = level
        self.path = os.path.join(path, "log.txt")
        with open(self.path, "a") as f:
            f.write(time.asctime() + "Start Logging \n")

    def __call__(self, type, msg):
        """
        Записать сообщение msg в файл self.path/log.txt
        """
        if type in self.level:
            with open(self.path, "a") as f:
                f.write(time.asctime() + " |-{}-| {} \n".format(type, msg))

    def read(self, file):
        """
        Получить записи из файла file.txt
        """
        with open(os.path.join(self.base_path, str(file) + ".txt"), "r") as f:
            text = f.read()
        return text

    def dump(self, arifact, name_artifact=None, add_path = ''):
        """
        Сохранить arifact self.path/name_artifact.pkl
        """
        if "tests_logs" in self.level:
            if name_artifact is None:
                name_artifact = str(arifact)
            with open(
                os.path.join(self.base_path, add_path, str(name_artifact) + ".pkl"), "wb"
            ) as f:
                pickle.dump(arifact, f)

    def save(self, data, name=None):
        """
        Сохранить DataFrame со специальным именем в специальную директорию self.path/results/name.xlsx.

        :param data: DataFrame to save
        :param name: filename
        """
        data = pd.DataFrame(data)
        if name is None:
            name = data.iloc[0].index[0]
        path = os.path.join(self.base_path, "results")
        if name[-5:] == ".xlsx":
            name = name[:-5]
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, name)
        data.to_excel(filename + ".xlsx")

    def load(self, name_artifact, drop_log=False, add_path = ''):
        """
        Получить arifact из self.path/name_artifact.pkl
        """
        try:
            with open(
                os.path.join(self.base_path, add_path, str(name_artifact) + ".pkl"), "rb"
            ) as f:
                arifact = pickle.load(f)
            return arifact
        except Exception as e:
            if not drop_log:
                self.__call__(
                    "valid_errors", "[Чтение log] Нет такого артефакта: " + str(e)
                )
            return None

    def get_all_logs(self, add_path=None, drop_log=False):
        """
        Вернуть словарь из всех залогированных элементов в директории self.path
        Читает только директории (как вложенный словарь), txt и pickle
        Если считать не получалось, то на месте элемента по ключу находится None
        """
        if ~os.path.exists(os.path.join(self.base_path, add_path)):
            return None

        if add_path is not None:
            files = os.listdir(os.path.join(self.base_path, add_path))
        else:
            files = os.listdir(self.base_path)
        artifacts = {}
        for file in files:
            if os.path.isdir(os.path.join(self.base_path, file)):
                artifacts[file] = self.get_all_logs(file, drop_log=True)
            else:
                file = ".".join(file.split(".")[:-1])
                try:
                    artifacts[file] = self.load(file, drop_log=True)
                    if artifacts[file] is None:
                        artifacts[file] = self.read(file)
                except Exception as e:
                    pass
        return artifacts

    def reset(self, path_to_clear=None):
        """
        Удалить все логи и артефакты, которые находятся в папке self.base_path
        """
        if path_to_clear is None:
            path_to_clear = self.base_path
        files = os.listdir(path_to_clear)
        for f in files:
            curr_path = os.path.join(path_to_clear, f)
            if os.path.isdir(curr_path):
                shutil.rmtree(curr_path)
            else:
                os.remove(curr_path)

    def checkOrCreate(self, add_path):
        '''
        Проверяет наличие директории. Если нет, то создает ее.
        '''
        if not os.path.exists(os.path.join(self.base_path, add_path)):
            os.makedirs(os.path.join(self.base_path, add_path))
