#!/usr/bin/env python3
"""
icon_vectorizer_gui.py

Простое окно: открыл растровую иконку -> увидел превью SVG -> сохранил файл.
Использует тот же движок конвертации, что и icon_vectorizer.py (должен лежать
в той же папке, либо код можно склеить в один файл — см. комментарий внизу).

Запуск:
    python3 icon_vectorizer_gui.py

Зависимости (кроме модулей самого движка):
    pip install pillow
    (tkinter уже входит в стандартную установку Python на Windows)

Опционально для drag&drop мышью прямо в окно:
    pip install tkinterdnd2
Если tkinterdnd2 не установлен, drag&drop просто отключается, а кнопка
"Открыть изображение" продолжает работать как обычно.
"""

import io
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- Двигатель конвертации (тот же код, что в icon_vectorizer.py) ---
try:
    from icon_vectorizer import vectorize, ICON_SIZE, FRAME_SIZE  # noqa: F401
except ImportError:
    messagebox.showerror(
        "Ошибка",
        "Не найден файл icon_vectorizer.py.\n"
        "Он должен лежать в той же папке, что и icon_vectorizer_gui.py.",
    )
    sys.exit(1)

try:
    import cairosvg  # для превью SVG прямо в окне
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


class IconVectorizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Icon Vectorizer")
        self.root.geometry("640x420")
        self.root.resizable(False, False)

        self.input_path = None
        self.output_svg_path = None

        # --- Верхняя панель кнопок ---
        top = tk.Frame(root, pady=10)
        top.pack(fill="x")

        tk.Button(top, text="Открыть изображение…", command=self.open_image, width=22).pack(side="left", padx=10)
        self.save_btn = tk.Button(top, text="Сохранить SVG…", command=self.save_svg, width=18, state="disabled")
        self.save_btn.pack(side="left", padx=10)

        self.threshold_var = tk.IntVar(value=127)
        tk.Label(top, text="Порог:").pack(side="left", padx=(20, 2))
        tk.Scale(top, from_=0, to=255, orient="horizontal", variable=self.threshold_var,
                 length=120, command=lambda _e: self.reconvert()).pack(side="left")

        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Инвертировать", variable=self.invert_var,
                        command=self.reconvert).pack(side="left", padx=10)

        # --- Область превью: оригинал слева, результат справа ---
        preview = tk.Frame(root)
        preview.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.LabelFrame(preview, text="Оригинал", width=300, height=300)
        left.pack(side="left", padx=10)
        left.pack_propagate(False)
        self.orig_label = tk.Label(left, text="Файл не выбран", fg="gray")
        self.orig_label.pack(expand=True)

        right = tk.LabelFrame(preview, text="Результат (SVG)", width=300, height=300)
        right.pack(side="left", padx=10)
        right.pack_propagate(False)
        self.result_label = tk.Label(right, text="—", fg="gray")
        self.result_label.pack(expand=True)

        self.status = tk.Label(root, text="Готов к работе.", anchor="w", bd=1, relief="sunken")
        self.status.pack(fill="x", side="bottom")

        # --- Drag & drop (опционально) ---
        self._setup_dnd()

    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa
            # Если используется TkinterDnD.Tk() как root — работает "из коробки".
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
            self.status.config(text="Готов к работе. Можно перетащить файл в окно.")
        except Exception:
            pass  # tkinterdnd2 не установлен — просто нет drag&drop, не критично

    def _on_drop(self, event):
        path = event.data.strip("{}")  # tkinterdnd2 иногда оборачивает путь в {}
        self._load_image(path)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Все файлы", "*.*")],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        self.input_path = path
        try:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((280, 280))
            self._orig_imgtk = ImageTk.PhotoImage(img)
            self.orig_label.config(image=self._orig_imgtk, text="")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{exc}")
            return

        self.status.config(text=f"Загружено: {os.path.basename(path)}")
        self.reconvert()

    def reconvert(self):
        if not self.input_path:
            return
        tmp_svg = os.path.join(os.path.dirname(self.input_path) or ".", "_icon_preview_tmp.svg")
        try:
            vectorize(
                self.input_path,
                tmp_svg,
                invert=self.invert_var.get(),
                threshold=self.threshold_var.get(),
            )
        except Exception as exc:
            self.result_label.config(image="", text=f"Ошибка:\n{exc}", fg="red")
            self.save_btn.config(state="disabled")
            self.status.config(text="Не удалось векторизовать изображение.")
            return

        self.output_svg_path = tmp_svg
        self._render_svg_preview(tmp_svg)
        self.save_btn.config(state="normal")
        self.status.config(text="Готово. Можно скорректировать порог или сохранить SVG.")

    def _render_svg_preview(self, svg_path):
        if HAS_CAIROSVG:
            try:
                png_bytes = cairosvg.svg2png(url=svg_path, scale=10, background_color="white")
                img = Image.open(io.BytesIO(png_bytes))
                img.thumbnail((280, 280))
                self._result_imgtk = ImageTk.PhotoImage(img)
                self.result_label.config(image=self._result_imgtk, text="")
                return
            except Exception:
                pass
        # Фолбэк, если cairosvg не установлен: просто сообщаем, что файл готов.
        self.result_label.config(
            image="", text="SVG сконвертирован.\n(для превью в окне\nустановите cairosvg)", fg="black"
        )

    def save_svg(self):
        if not self.output_svg_path or not os.path.exists(self.output_svg_path):
            return
        default_name = os.path.splitext(os.path.basename(self.input_path))[0] + ".svg"
        path = filedialog.asksaveasfilename(
            title="Сохранить SVG",
            defaultextension=".svg",
            initialfile=default_name,
            filetypes=[("SVG файл", "*.svg")],
        )
        if not path:
            return
        with open(self.output_svg_path, "r", encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        self.status.config(text=f"Сохранено: {path}")
        messagebox.showinfo("Готово", f"SVG сохранён:\n{path}")


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()

    app = IconVectorizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
