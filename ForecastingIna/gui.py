from doctest import master
from tkinter import *
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from pathlib import Path
from tkinter import ttk
import pandas as pd


def main():
    master = Tk()
    master.title('UniFree Demand Forecasting')
    canvas = Canvas(master, height=450, width=750)
    canvas.pack()

    frame_ust = Frame(master, bg='light blue')
    frame_ust.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)

    resim = ImageTk.PhotoImage(Image.open('brand.png'))
    frame_altSol = Label(master, bg='light blue', image=resim)
    frame_altSol.place(relx=0.1, rely=0.21, relwidth=0.23, relheight=0.5)

    frame_altSag = Frame(master, bg='light blue')
    frame_altSag.place(relx=0.34, rely=0.21, relwidth=0.56, relheight=0.5)

    hatirlatma_tipi_etiket = Label(frame_ust, bg='light blue', text="Demand Forecasting:", font='Verdana 12 bold')
    hatirlatma_tipi_etiket.pack(padx=10, pady=10, side=LEFT)

    folderSelection = Label(frame_altSag, bg='light blue', text='Lütfen tahminlenmiş veriyi giriniz.',
                            font='Verdana 12 bold')
    folderSelection.pack(padx=10, pady=10, side=TOP)
    yazi = Label(frame_altSag, bg='light blue')
    yazi.config(text="Tahmin edileni seçme ve incelemek için tıkla.")
    yazi.pack(padx=10, pady=10, side=TOP)

    buton = Label(frame_altSag, bg='light blue')
    buton.config(text="")
    buton.place(relx=0.4, rely=0.4, relwidth=0.15, relheight=0.1)

    def degistir():
        def drop(event):
            var.set(event.data)

        class Application(TkinterDnD.Tk):
            def __init__(self):
                super().__init__()
                self.title("Detaylı görüntüleme arayüzü")
                self.main_frame = tk.Frame(self)
                self.main_frame.pack(fill="both", expand="true")
                self.geometry("900x500")
                self.search_page = SearchPage(parent=self.main_frame)

        class DataTable(ttk.Treeview):
            def __init__(self, parent):
                super().__init__(parent)
                scroll_Y = tk.Scrollbar(self, orient="vertical", command=self.yview)
                scroll_X = tk.Scrollbar(self, orient="horizontal", command=self.xview)
                self.configure(yscrollcommand=scroll_Y.set, xscrollcommand=scroll_X.set)
                scroll_Y.pack(side="right", fill="y")
                scroll_X.pack(side="bottom", fill="x")
                self.stored_dataframe = pd.DataFrame()

            def set_datatable(self, dataframe):
                self.stored_dataframe = dataframe
                self._draw_table(dataframe)

            def _draw_table(self, dataframe):
                self.delete(*self.get_children())
                columns = list(dataframe.columns)
                self.__setitem__("column", columns)
                self.__setitem__("show", "headings")

                for col in columns:
                    self.heading(col, text=col)

                df_rows = dataframe.to_numpy().tolist()
                for row in df_rows:
                    self.insert("", "end", values=row)
                return None

            def find_value(self, pairs):
                # pairs is a dictionary
                new_df = self.stored_dataframe
                for col, value in pairs.items():
                    query_string = f"{col}.str.contains('{value}')"
                    new_df = new_df.query(query_string, engine="python")
                self._draw_table(new_df)

            def reset_table(self):
                self._draw_table(self.stored_dataframe)

        class SearchPage(tk.Frame):
            def __init__(self, parent):
                super().__init__(parent)

                self.file_names_listbox = tk.Listbox(parent, selectmode=tk.SINGLE,
                                                     background="light blue")
                self.file_names_listbox.place(relheight=1, relwidth=0.25)
                self.file_names_listbox.drop_target_register(DND_FILES)
                self.file_names_listbox.dnd_bind("<<Drop>>", self.drop_inside_list_box)
                self.file_names_listbox.bind("<Double-1>", self._display_file)

                self.search_entrybox = tk.Entry(parent)
                self.search_entrybox.place(relx=0.25, relwidth=0.75)
                self.search_entrybox.bind("<Return>", self.search_table)

                # Treeview
                self.data_table = DataTable(parent)
                self.data_table.place(rely=0.05, relx=0.25, relwidth=0.75, relheight=0.95)

                self.path_map = {}

            def drop_inside_list_box(self, event):
                file_paths = self._parse_drop_files(event.data)
                current_listbox_items = set(self.file_names_listbox.get(0, "end"))
                for file_path in file_paths:
                    if file_path.endswith(".csv"):
                        path_object = Path(file_path)
                        file_name = path_object.name
                        if file_name not in current_listbox_items:
                            self.file_names_listbox.insert("end", file_name)
                            self.path_map[file_name] = file_path

            def _display_file(self, event):
                file_name = self.file_names_listbox.get(self.file_names_listbox.curselection())
                path = self.path_map[file_name]
                df = pd.read_csv(path)
                self.data_table.set_datatable(dataframe=df)

            def _parse_drop_files(self, filename):
                size = len(filename)
                res = []  # list of file paths
                name = ""
                idx = 0
                while idx < size:
                    if filename[idx] == "{":
                        j = idx + 1
                        while filename[j] != "}":
                            name += filename[j]
                            j += 1
                        res.append(name)
                        name = ""
                        idx = j
                    elif filename[idx] == " " and name != "":
                        res.append(name)
                        name = ""
                    elif filename[idx] != " ":
                        name += filename[idx]
                    idx += 1
                if name != "":
                    res.append(name)
                return res

            def search_table(self, event):
                # column value. [[column,value],column2=value2]....
                entry = self.search_entrybox.get()
                if entry == "":
                    self.data_table.reset_table()
                else:
                    entry_split = entry.split(",")
                    column_value_pairs = {}
                    for pair in entry_split:
                        pair_split = pair.split("=")
                        if len(pair_split) == 2:
                            col = pair_split[0]
                            lookup_value = pair_split[1]
                            column_value_pairs[col] = lookup_value
                    self.data_table.find_value(pairs=column_value_pairs)

        if __name__ == "__main__":
            root = Application()
            root.mainloop()

    dugme = tk.Button(buton)
    dugme.config(text="select", bg='grey')
    dugme.config(command=degistir)
    dugme.place(relx=0.34, rely=0.21, relwidth=0.8, relheight=1)
    mainloop()
    master.mainloop()


main()

'''import PySimpleGUI as sg
from io import StringIO

import pandas as pd

sg.theme("DarkTeal2")
layout = [
    [sg.T("")],
    [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
    [sg.Button("Submit")]
]

window = sg.Window('My File Browser', layout, size=(600, 150))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == ''Submit'':
        buffer = StringIO(values["-IN-"])
        data = pd.read_csv(buffer, sep=";")
window.close()
'''
