from dearpygui.dearpygui import *

create_context()

# Global state
input_controls = []
hidden_layer_controls = []
output_displays = []

def update_input_fields(sender, app_data, user_data):
    count = get_value("##num_inputs")
    for ctrl in input_controls:
        delete_item(ctrl)
    input_controls.clear()

    for i in range(count):
        label = f"x{i+1}"
        input_id = add_input_float(label=label, default_value=0.0, width=100, parent="input_container")
        input_controls.append(input_id)

def update_hidden_layer_fields(sender, app_data, user_data):
    count = get_value("##num_hidden")
    for ctrl in hidden_layer_controls:
        delete_item(ctrl)
    hidden_layer_controls.clear()

    for i in range(count):
        label = f"Neurons in Hidden Layer {i+1}"
        spin_id = add_input_int(label=label, default_value=1, min_value=1, width=100, parent="hidden_container")
        hidden_layer_controls.append(spin_id)

def update_output_fields(sender, app_data, user_data):
    count = get_value("##num_outputs")
    for disp in output_displays:
        delete_item(disp)
    output_displays.clear()

    for i in range(count):
        label = f"y{i+1}"
        display_id = add_text(label + ": (pending)", parent="output_container")
        output_displays.append(display_id)

def calculate_outputs(sender, app_data, user_data):
    try:
        inputs = [get_value(ctrl) for ctrl in input_controls]
        hidden_layers = [get_value(ctrl) for ctrl in hidden_layer_controls]
        num_outputs = get_value("##num_outputs")

        results = []
        for i in range(num_outputs):
            sum_inputs = sum(inputs)
            result = (sum_inputs + i) / (len(hidden_layers) + 1.0)
            results.append(result)

        for i, val in enumerate(results):
            set_value(output_displays[i], f"y{i+1}: {val:.4f}")
    except Exception as e:
        print("Calculation error:", e)

# Main window
with window(label="DNN Configurator", width=500, height=600):
    add_text("🔢 Input Layer Configuration")
    add_input_int(label="Number of Inputs", default_value=2, min_value=0, tag="##num_inputs", callback=update_input_fields)
    add_group(horizontal=False, tag="input_container")

    add_separator()
    add_text("🧠 Hidden Layers")
    add_input_int(label="Number of Hidden Layers", default_value=1, min_value=0, tag="##num_hidden", callback=update_hidden_layer_fields)
    add_group(horizontal=False, tag="hidden_container")

    add_separator()
    add_text("🎯 Output Layer")
    add_input_int(label="Number of Outputs", default_value=1, min_value=0, tag="##num_outputs", callback=update_output_fields)
    add_group(horizontal=False, tag="output_container")

    add_separator()
    add_button(label="Calculate", callback=calculate_outputs)

# Trigger initial UI setup
set_value("##num_inputs", 2)
update_input_fields(None, None, None)
set_value("##num_hidden", 1)
update_hidden_layer_fields(None, None, None)
set_value("##num_outputs", 1)
update_output_fields(None, None, None)

# Run the app
create_viewport(title='Neural Network Configurator', width=520, height=700)
setup_dearpygui()
show_viewport()
start_dearpygui()
destroy_context()
