from dearpygui.dearpygui import *

def hello_callback():
    print("Merhaba! Butona tıkladın.")

create_context()  # BU EN ÖNCE OLMALI

with window(label="Basit GUI", width=400, height=200):
    add_text("👋 Merhaba DNN Dünyası!")
    add_button(label="Tıkla", callback=hello_callback)

create_viewport(title="Test GUI", width=420, height=240)
setup_dearpygui()
show_viewport()
start_dearpygui()
destroy_context()
