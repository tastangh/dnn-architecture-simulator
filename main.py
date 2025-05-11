import dearpygui.dearpygui as dpg
import numpy as np
import uuid # For unique tags

# --- Constants and Global State ---
ID_PRIMARY_WINDOW = "primary_window"
ID_STATUS_TEXT = "status_text"

# Network structure and parameters
network_config = {
    "num_inputs": 2,
    "num_hidden_layers": 1,
    "hidden_layer_neurons": [3], # Neurons in each hidden layer
    "num_outputs": 1,
    "learning_rate": 0.01
}

# Will store numpy arrays for weights, biases, gradients, activations
# and DPG tags for their UI elements
network_params_ui = {
    "weights": [],  # List of [UI_group_tag, numpy_array, {neuron_from_tag: {neuron_to_tag: weight_input_tag}}]
    "biases": [],   # List of [UI_group_tag, numpy_array, {neuron_tag: bias_input_tag}]
    "layer_activations_z": [], # numpy arrays
    "layer_activations_a": [], # numpy arrays
    "gradients_w": [], # numpy arrays
    "gradients_b": [], # numpy arrays
    "activation_fn_choices": [], # DPG combo tags for activation functions per layer
    "input_value_tags": [],
    "target_value_tags": [],
    "output_display_tags": [],
    "loss_display_tag": None,
    "gradient_display_tags_w": [], # Mirror structure of weights UI
    "gradient_display_tags_b": []  # Mirror structure of biases UI
}

ACTIVATION_FUNCTIONS = ["Linear", "Sigmoid", "ReLU", "Tanh"]
LOSS_FUNCTIONS = ["Mean Squared Error"] # Add more later if needed

# --- Helper Functions ---

def get_activation_function(name):
    if name == "Sigmoid":
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == "ReLU":
        return lambda x: np.maximum(0, x)
    elif name == "Tanh":
        return lambda x: np.tanh(x)
    elif name == "Linear":
        return lambda x: x
    return lambda x: x # Default to Linear

def get_activation_derivative(name):
    # Note: These derivatives expect the output of the activation function (a),
    # or z for some implementations. Here, let's assume 'a' for sigmoid/tanh, 'z' for relu.
    if name == "Sigmoid":
        return lambda a: a * (1 - a)
    elif name == "ReLU": # Derivative w.r.t z
        return lambda z: (z > 0).astype(float)
    elif name == "Tanh": # Derivative w.r.t a
        return lambda a: 1 - a**2
    elif name == "Linear":
        return lambda z_or_a: np.ones_like(z_or_a)
    return lambda z_or_a: np.ones_like(z_or_a)

def _clear_dpg_items_from_list_of_tags(tag_list):
    for tag in tag_list:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    tag_list.clear()

def _clear_dpg_items_from_nested_dict(nested_dict_tags):
    if isinstance(nested_dict_tags, dict):
        for key, value in nested_dict_tags.items():
            _clear_dpg_items_from_nested_dict(value) # Recurse for nested dicts
            if isinstance(key, (str, int)) and dpg.does_item_exist(key) and not dpg.is_item_container(key):
                dpg.delete_item(key) # Delete if key itself is a tag
            if isinstance(value, (str, int)) and dpg.does_item_exist(value) and not dpg.is_item_container(value):
                dpg.delete_item(value) # Delete if value is a tag
    elif isinstance(nested_dict_tags, list):
         for item in nested_dict_tags:
            _clear_dpg_items_from_list_of_tags(item)


def _clear_network_params_ui_elements():
    """Clears DPG elements related to weights, biases, gradients, IO etc."""
    global network_params_ui
    
    for w_entry in network_params_ui["weights"]: # [UI_group_tag, np_array, {from: {to: tag}}]
        if dpg.does_item_exist(w_entry[0]): dpg.delete_item(w_entry[0]) # Delete group
        # Individual input fields are children, so they get deleted with the group
    for b_entry in network_params_ui["biases"]: # [UI_group_tag, np_array, {neuron: tag}]
        if dpg.does_item_exist(b_entry[0]): dpg.delete_item(b_entry[0])

    for grad_w_group_tag in network_params_ui["gradient_display_tags_w"]:
        if dpg.does_item_exist(grad_w_group_tag): dpg.delete_item(grad_w_group_tag)
    for grad_b_group_tag in network_params_ui["gradient_display_tags_b"]:
        if dpg.does_item_exist(grad_b_group_tag): dpg.delete_item(grad_b_group_tag)

    _clear_dpg_items_from_list_of_tags(network_params_ui["activation_fn_choices"])
    _clear_dpg_items_from_list_of_tags(network_params_ui["input_value_tags"])
    _clear_dpg_items_from_list_of_tags(network_params_ui["target_value_tags"])
    _clear_dpg_items_from_list_of_tags(network_params_ui["output_display_tags"])
    
    if network_params_ui["loss_display_tag"] and dpg.does_item_exist(network_params_ui["loss_display_tag"]):
        dpg.delete_item(network_params_ui["loss_display_tag"])
    
    # Reset the storage
    network_params_ui = {
        "weights": [], "biases": [], "layer_activations_z": [], "layer_activations_a": [],
        "gradients_w": [], "gradients_b": [], "activation_fn_choices": [],
        "input_value_tags": [], "target_value_tags": [], "output_display_tags": [],
        "loss_display_tag": None, "gradient_display_tags_w": [], "gradient_display_tags_b": []
    }
    # Clear parent groups too
    if dpg.does_item_exist("weights_biases_group"):
        dpg.delete_item("weights_biases_group", children_only=True)
    if dpg.does_item_exist("activations_group"):
        dpg.delete_item("activations_group", children_only=True)
    if dpg.does_item_exist("io_values_group"):
        dpg.delete_item("io_values_group", children_only=True)
    if dpg.does_item_exist("results_group"):
        dpg.delete_item("results_group", children_only=True)
    if dpg.does_item_exist("gradients_display_group"):
         dpg.delete_item("gradients_display_group", children_only=True)

# --- UI Building Functions ---

def update_hidden_layer_neuron_inputs(sender, app_data, user_data):
    """Dynamically creates input_int for neuron counts in each hidden layer."""
    num_hidden = dpg.get_value("num_hidden_layers_spin")
    network_config["num_hidden_layers"] = num_hidden
    
    # Clear old inputs
    if dpg.does_item_exist("hidden_layer_neurons_group"):
        dpg.delete_item("hidden_layer_neurons_group", children_only=True)
    
    network_config["hidden_layer_neurons"] = [] # Reset
    
    for i in range(num_hidden):
        default_val = 1
        if i < len(network_config["hidden_layer_neurons"]): # Preserve old values if possible
             default_val = network_config["hidden_layer_neurons"][i]

        tag = f"hidden_neurons_spin_{i}"
        dpg.add_input_int(label=f"Neurons in Hidden Layer {i+1}",
                          default_value=default_val, min_value=1, tag=tag,
                          parent="hidden_layer_neurons_group", width=100,
                          callback=lambda s,a,u,idx=i: update_specific_hidden_neuron_count(idx, a))
        if i >= len(network_config["hidden_layer_neurons"]):
            network_config["hidden_layer_neurons"].append(default_val)
        else:
            network_config["hidden_layer_neurons"][i] = default_val


def update_specific_hidden_neuron_count(layer_index, count):
    if layer_index < len(network_config["hidden_layer_neurons"]):
        network_config["hidden_layer_neurons"][layer_index] = count
    else: # Should not happen if UI is built correctly
        print(f"Error: Trying to update hidden neuron count for out-of-bounds index {layer_index}")


def build_network_structure_and_ui(sender=None, app_data=None, user_data=None):
    """Defines network structure and builds UI for weights, biases, activations."""
    global network_params_ui
    dpg.set_value(ID_STATUS_TEXT, "Building network structure...")
    _clear_network_params_ui_elements()

    # Get latest values from architecture definition
    network_config["num_inputs"] = dpg.get_value("num_inputs_spin")
    network_config["num_outputs"] = dpg.get_value("num_outputs_spin")
    # hidden_layer_neurons should be up-to-date via callbacks

    layer_sizes = [network_config["num_inputs"]] + \
                  network_config["hidden_layer_neurons"] + \
                  [network_config["num_outputs"]]

    # --- Create UI for Weights and Biases ---
    current_weights_group = dpg.add_group(parent="weights_biases_group", horizontal=False)
    
    for i in range(len(layer_sizes) - 1):
        rows, cols = layer_sizes[i+1], layer_sizes[i] # Output neurons, Input neurons
        
        # Weights W_ij from neuron j (prev layer) to neuron i (current layer)
        # So matrix shape is (neurons_curr_layer, neurons_prev_layer)
        # Initialize with small random numbers
        weights_np = np.random.randn(rows, cols) * 0.1 
        biases_np = np.zeros(rows) # Bias for each neuron in current layer (i+1)

        # UI for Weights for this layer connection
        layer_weight_group_tag = f"weights_layer_{i}_to_{i+1}_group"
        dpg.add_text(f"Weights: Layer {i} ({cols} neurons) to Layer {i+1} ({rows} neurons)", parent=current_weights_group)
        ui_w_group = dpg.add_group(tag=layer_weight_group_tag, parent=current_weights_group, horizontal=False)
        weight_tags_for_layer = {}
        for r_idx in range(rows): # Neuron in current layer (destination)
            row_tag_group = dpg.add_group(parent=ui_w_group, horizontal=True)
            dpg.add_text(f"  To N{r_idx+1}: ", parent=row_tag_group)
            weight_tags_for_layer[r_idx] = {}
            for c_idx in range(cols): # Neuron in previous layer (source)
                w_tag = f"w_l{i}_n{c_idx}_to_n{r_idx}"
                dpg.add_input_float(tag=w_tag, default_value=weights_np[r_idx, c_idx], 
                                    width=70, format="%.3f", parent=row_tag_group)
                weight_tags_for_layer[r_idx][c_idx] = w_tag
        network_params_ui["weights"].append([layer_weight_group_tag, weights_np, weight_tags_for_layer])
        dpg.add_separator(parent=current_weights_group)

        # UI for Biases for layer i+1 (not for input layer)
        layer_bias_group_tag = f"biases_layer_{i+1}_group"
        dpg.add_text(f"Biases: Layer {i+1} ({rows} neurons)", parent=current_weights_group)
        ui_b_group = dpg.add_group(tag=layer_bias_group_tag, parent=current_weights_group, horizontal=True)
        bias_tags_for_layer = {}
        for r_idx in range(rows):
            b_tag = f"b_l{i+1}_n{r_idx}"
            dpg.add_input_float(tag=b_tag, default_value=biases_np[r_idx], 
                                width=70, format="%.3f", parent=ui_b_group)
            bias_tags_for_layer[r_idx] = b_tag
        network_params_ui["biases"].append([layer_bias_group_tag, biases_np, bias_tags_for_layer])
        dpg.add_separator(parent=current_weights_group)
        
        # UI for Activation Function for layer i+1
        act_tag = f"activation_l{i+1}_combo"
        dpg.add_combo(ACTIVATION_FUNCTIONS, label=f"Activation L{i+1}", default_value="Sigmoid", 
                      tag=act_tag, parent="activations_group")
        network_params_ui["activation_fn_choices"].append(act_tag)

    # --- Create UI for Input Values ---
    dpg.add_text("Input Values (x_i):", parent="io_values_group")
    input_group = dpg.add_group(parent="io_values_group", horizontal=True)
    for i in range(network_config["num_inputs"]):
        tag = f"input_x{i}"
        dpg.add_input_float(label=f"x{i+1}", tag=tag, default_value=0.0, width=80, parent=input_group)
        network_params_ui["input_value_tags"].append(tag)
    dpg.add_separator(parent="io_values_group")

    # --- Create UI for Target Values ---
    dpg.add_text("Target Values (y_true_i):", parent="io_values_group")
    target_group = dpg.add_group(parent="io_values_group", horizontal=True)
    for i in range(network_config["num_outputs"]):
        tag = f"target_y{i}"
        dpg.add_input_float(label=f"yt{i+1}", tag=tag, default_value=0.0, width=80, parent=target_group)
        network_params_ui["target_value_tags"].append(tag)

    # --- Create UI for Calculated Outputs & Loss ---
    dpg.add_text("Calculated Outputs (y_pred_i):", parent="results_group")
    output_disp_group = dpg.add_group(parent="results_group", horizontal=True)
    for i in range(network_config["num_outputs"]):
        tag = f"output_display_y{i}"
        dpg.add_text(f"yp{i+1}: (calc)", tag=tag, parent=output_disp_group)
        network_params_ui["output_display_tags"].append(tag)
    
    loss_tag = "loss_display_text"
    dpg.add_text("Loss: (calc)", tag=loss_tag, parent="results_group")
    network_params_ui["loss_display_tag"] = loss_tag
    
    # --- Create UI for Gradients (placeholders, values updated after backward pass) ---
    dpg.add_text("Calculated Gradients:", parent="gradients_display_group")
    for i in range(len(layer_sizes) - 1): # For each weight matrix
        rows, cols = layer_sizes[i+1], layer_sizes[i]
        grad_w_group_tag = f"grad_w_layer_{i}_to_{i+1}_group"
        dpg.add_text(f"Grad-Weights: L{i} to L{i+1}", parent="gradients_display_group")
        ui_gw_group = dpg.add_group(tag=grad_w_group_tag, parent="gradients_display_group", horizontal=False)
        grad_w_tags_for_layer = {}
        for r_idx in range(rows):
            row_tag_group = dpg.add_group(parent=ui_gw_group, horizontal=True)
            dpg.add_text(f"  To N{r_idx+1}: ", parent=row_tag_group)
            grad_w_tags_for_layer[r_idx] = {}
            for c_idx in range(cols):
                gw_tag = f"grad_w_l{i}_n{c_idx}_to_n{r_idx}_display"
                dpg.add_text("0.000", tag=gw_tag, parent=row_tag_group) # Placeholder
                grad_w_tags_for_layer[r_idx][c_idx] = gw_tag
        network_params_ui["gradient_display_tags_w"].append([grad_w_group_tag, grad_w_tags_for_layer])
        dpg.add_separator(parent="gradients_display_group")

        grad_b_group_tag = f"grad_b_layer_{i+1}_group"
        dpg.add_text(f"Grad-Biases: L{i+1}", parent="gradients_display_group")
        ui_gb_group = dpg.add_group(tag=grad_b_group_tag, parent="gradients_display_group", horizontal=True)
        grad_b_tags_for_layer = {}
        for r_idx in range(rows):
            gb_tag = f"grad_b_l{i+1}_n{r_idx}_display"
            dpg.add_text("0.000", tag=gb_tag, parent=ui_gb_group) # Placeholder
            grad_b_tags_for_layer[r_idx] = gb_tag
        network_params_ui["gradient_display_tags_b"].append([grad_b_group_tag, grad_b_tags_for_layer])
        dpg.add_separator(parent="gradients_display_group")

    dpg.set_value(ID_STATUS_TEXT, "Network structure and UI built. Enter parameters.")
    # Make sure the window resizes if content is large
    dpg.configure_item(ID_PRIMARY_WINDOW, width=dpg.get_item_width(ID_PRIMARY_WINDOW), height=dpg.get_item_height(ID_PRIMARY_WINDOW))


def collect_parameters_from_ui():
    """Reads weights, biases, learning rate from UI into numpy arrays."""
    global network_params_ui, network_config

    network_config["learning_rate"] = dpg.get_value("learning_rate_input")

    for i, (group_tag, weights_np, weight_tags_map) in enumerate(network_params_ui["weights"]):
        for r_idx, row_map in weight_tags_map.items():
            for c_idx, tag in row_map.items():
                weights_np[r_idx, c_idx] = dpg.get_value(tag)
    
    for i, (group_tag, biases_np, bias_tags_map) in enumerate(network_params_ui["biases"]):
        for r_idx, tag in bias_tags_map.items():
            biases_np[r_idx] = dpg.get_value(tag)
    
    dpg.set_value(ID_STATUS_TEXT, "Parameters collected from UI.")

def update_parameters_in_ui():
    """Updates weight and bias input fields in the UI from numpy arrays."""
    for i, (group_tag, weights_np, weight_tags_map) in enumerate(network_params_ui["weights"]):
        for r_idx, row_map in weight_tags_map.items():
            for c_idx, tag in row_map.items():
                dpg.set_value(tag, float(weights_np[r_idx, c_idx]))
    
    for i, (group_tag, biases_np, bias_tags_map) in enumerate(network_params_ui["biases"]):
        for r_idx, tag in bias_tags_map.items():
            dpg.set_value(tag, float(biases_np[r_idx]))
    dpg.set_value(ID_STATUS_TEXT, "Weights and Biases UI updated.")


# --- Neural Network Operations ---

def forward_pass_callback(sender, app_data, user_data):
    if not network_params_ui["weights"]: # Check if network is built
        dpg.set_value(ID_STATUS_TEXT, "Error: Network structure not built yet.")
        return
    collect_parameters_from_ui() # Ensure current params are used
    dpg.set_value(ID_STATUS_TEXT, "Performing forward pass...")

    network_params_ui["layer_activations_z"] = []
    network_params_ui["layer_activations_a"] = []

    # Get input values
    current_a = np.array([dpg.get_value(tag) for tag in network_params_ui["input_value_tags"]])
    network_params_ui["layer_activations_a"].append(current_a) # Store input layer activations

    num_layers = len(network_params_ui["weights"]) # Number of weight matrices = number of layers with weights/biases

    for i in range(num_layers):
        weights = network_params_ui["weights"][i][1] # weights_np
        biases = network_params_ui["biases"][i][1]   # biases_np
        activation_choice_tag = network_params_ui["activation_fn_choices"][i]
        activation_name = dpg.get_value(activation_choice_tag)
        act_fn = get_activation_function(activation_name)

        # z = W * a_prev + b
        current_z = np.dot(weights, current_a) + biases
        current_a = act_fn(current_z)
        
        network_params_ui["layer_activations_z"].append(current_z)
        network_params_ui["layer_activations_a"].append(current_a) # Store for layer i+1

    # Display final outputs
    final_outputs = network_params_ui["layer_activations_a"][-1]
    for i, tag in enumerate(network_params_ui["output_display_tags"]):
        dpg.set_value(tag, f"yp{i+1}: {final_outputs[i]:.4f}")
    
    dpg.set_value(ID_STATUS_TEXT, "Forward pass complete. Outputs calculated.")


def calculate_loss_callback(sender, app_data, user_data):
    if not network_params_ui["layer_activations_a"] or \
       len(network_params_ui["layer_activations_a"]) < 2: # Need at least input and one output layer activation
        dpg.set_value(ID_STATUS_TEXT, "Error: Perform forward pass first.")
        return
    
    dpg.set_value(ID_STATUS_TEXT, "Calculating loss...")
    y_pred = network_params_ui["layer_activations_a"][-1]
    y_true = np.array([dpg.get_value(tag) for tag in network_params_ui["target_value_tags"]])
    
    loss_fn_name = dpg.get_value("loss_function_combo")
    loss = 0
    if loss_fn_name == "Mean Squared Error":
        # Ensure y_pred and y_true are same shape for element-wise operations
        if y_pred.shape != y_true.shape:
            dpg.set_value(ID_STATUS_TEXT, f"Error: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
            return
        loss = 0.5 * np.sum((y_pred - y_true)**2) # 0.5 factor common for easier derivative
    else:
        dpg.set_value(ID_STATUS_TEXT, f"Error: Loss function '{loss_fn_name}' not implemented.")
        return

    if network_params_ui["loss_display_tag"]:
        dpg.set_value(network_params_ui["loss_display_tag"], f"Loss: {loss:.6f}")
    dpg.set_value(ID_STATUS_TEXT, f"Loss calculated: {loss:.6f}")


def backward_pass_callback(sender, app_data, user_data):
    if not network_params_ui["layer_activations_z"]: # z values are crucial for derivatives
        dpg.set_value(ID_STATUS_TEXT, "Error: Perform forward pass first.")
        return

    y_pred = network_params_ui["layer_activations_a"][-1]
    y_true = np.array([dpg.get_value(tag) for tag in network_params_ui["target_value_tags"]])
    
    network_params_ui["gradients_w"] = [np.zeros_like(w[1]) for w in network_params_ui["weights"]]
    network_params_ui["gradients_b"] = [np.zeros_like(b[1]) for b in network_params_ui["biases"]]

    # --- Output Layer Error (delta L) ---
    # Assumes MSE loss derivative: (y_pred - y_true)
    loss_deriv = y_pred - y_true 
    
    output_layer_idx = len(network_params_ui["weights"]) - 1 # Index for the last set of weights/biases/activations
    
    # Activation derivative for output layer
    # z_L = network_params_ui["layer_activations_z"][-1] # z for the output layer
    # a_L = y_pred # a for the output layer
    
    act_choice_tag_output = network_params_ui["activation_fn_choices"][output_layer_idx] # Last activation choice
    act_name_output = dpg.get_value(act_choice_tag_output)
    act_deriv_fn_output = get_activation_derivative(act_name_output)

    # For ReLU, derivative needs z. For Sigmoid/Tanh, can use a.
    # Let's try to be consistent and use z if available, otherwise a (might need adjustment)
    # The derivative of sigmoid f'(z) = f(z)(1-f(z)) = a(1-a)
    # The derivative of ReLU f'(z) = 1 if z>0 else 0
    # The derivative of Tanh f'(z) = 1 - (f(z))^2 = 1 - a^2
    
    # Get z for the output layer (it's the last one in the list)
    z_output = network_params_ui["layer_activations_z"][-1]
    
    if act_name_output == "ReLU":
         act_derivative_output = act_deriv_fn_output(z_output)
    else: # Sigmoid, Tanh, Linear (using 'a' for sigmoid/tanh, 'z' for linear)
         act_derivative_output = act_deriv_fn_output(y_pred if act_name_output != "Linear" else z_output)


    delta = loss_deriv * act_derivative_output # Element-wise product

    # --- Gradients for the last layer (Output layer) ---
    # grad_w_L = delta_L (outer_product) a_{L-1}
    # grad_b_L = delta_L
    
    # a_{L-1} is the activation of the layer *before* the output layer.
    # If only one layer (input -> output), a_{L-1} is the input.
    # network_params_ui["layer_activations_a"] stores [input_a, hidden1_a, ..., output_a]
    # So, a_{L-1} is at index -2 of layer_activations_a
    a_prev = network_params_ui["layer_activations_a"][-2] 
    
    network_params_ui["gradients_w"][-1] = np.outer(delta, a_prev)
    network_params_ui["gradients_b"][-1] = delta

    # --- Propagate error backwards for hidden layers ---
    # Iterate from L-1 down to the first hidden layer
    for l_idx in range(len(network_params_ui["weights"]) - 2, -1, -1):
        # l_idx refers to the index of W, b, and activation_choice for the *current* layer
        # (e.g., if 3 layers of weights, l_idx goes 1, 0)
        # delta is from layer l+1 (the one closer to output, already computed)
        
        weights_next_layer = network_params_ui["weights"][l_idx+1][1] # W_{l+1}
        
        z_current_layer = network_params_ui["layer_activations_z"][l_idx]
        a_current_layer = network_params_ui["layer_activations_a"][l_idx+1] # a for current layer (index +1 because input is 0)

        act_choice_tag_curr = network_params_ui["activation_fn_choices"][l_idx]
        act_name_curr = dpg.get_value(act_choice_tag_curr)
        act_deriv_fn_curr = get_activation_derivative(act_name_curr)
        
        if act_name_curr == "ReLU":
             act_derivative_curr = act_deriv_fn_curr(z_current_layer)
        else:
             act_derivative_curr = act_deriv_fn_curr(a_current_layer if act_name_curr != "Linear" else z_current_layer)

        delta = np.dot(weights_next_layer.T, delta) * act_derivative_curr

        a_prev_for_current_grad = network_params_ui["layer_activations_a"][l_idx] # a from layer l-1
        network_params_ui["gradients_w"][l_idx] = np.outer(delta, a_prev_for_current_grad)
        network_params_ui["gradients_b"][l_idx] = delta

    # --- Display Gradients ---
    for i, (group_tag, grad_w_map) in enumerate(network_params_ui["gradient_display_tags_w"]):
        current_grad_w_np = network_params_ui["gradients_w"][i]
        for r_idx, row_map in grad_w_map.items():
            for c_idx, tag in row_map.items():
                dpg.set_value(tag, f"{current_grad_w_np[r_idx, c_idx]:.3f}")

    for i, (group_tag, grad_b_map) in enumerate(network_params_ui["gradient_display_tags_b"]):
        current_grad_b_np = network_params_ui["gradients_b"][i]
        for r_idx, tag in grad_b_map.items():
            dpg.set_value(tag, f"{current_grad_b_np[r_idx]:.3f}")
            
    dpg.set_value(ID_STATUS_TEXT, "Backward pass complete. Gradients calculated and displayed.")


def update_parameters_callback(sender, app_data, user_data):
    if not network_params_ui["gradients_w"]:
        dpg.set_value(ID_STATUS_TEXT, "Error: Calculate gradients (backward pass) first.")
        return
    
    lr = network_config["learning_rate"]
    dpg.set_value(ID_STATUS_TEXT, f"Updating parameters with LR={lr}...")

    for i in range(len(network_params_ui["weights"])):
        network_params_ui["weights"][i][1] -= lr * network_params_ui["gradients_w"][i]
        network_params_ui["biases"][i][1] -= lr * network_params_ui["gradients_b"][i]
        
    update_parameters_in_ui() # Reflect changes in the input fields
    dpg.set_value(ID_STATUS_TEXT, "Parameters updated.")

# --- Main DPG Setup ---
dpg.create_context()

with dpg.window(label="Neural Network Simulator", tag=ID_PRIMARY_WINDOW, width=1000, height=800):
    dpg.add_text("Neural Network Step-by-Step Simulator", color=(255, 255, 0))
    dpg.add_separator()

    # --- Architecture Definition ---
    with dpg.collapsing_header(label="1. Network Architecture", default_open=True):
        dpg.add_input_int(label="Input Neurons", tag="num_inputs_spin", default_value=network_config["num_inputs"], min_value=1, width=100)
        dpg.add_input_int(label="Number of Hidden Layers", tag="num_hidden_layers_spin", 
                          default_value=network_config["num_hidden_layers"], min_value=0, width=100,
                          callback=update_hidden_layer_neuron_inputs)
        dpg.add_group(tag="hidden_layer_neurons_group", horizontal=False) # Dynamic neuron counts here
        dpg.add_input_int(label="Output Neurons", tag="num_outputs_spin", default_value=network_config["num_outputs"], min_value=1, width=100)
        dpg.add_button(label="Build/Rebuild Network Structure & UI", callback=build_network_structure_and_ui, width=-1)
    dpg.add_separator()
    
    # Initial call to populate hidden layer neuron inputs based on default hidden layer count
    update_hidden_layer_neuron_inputs(None, network_config["num_hidden_layers"], None)


    # --- Parameters & Configuration (Dynamic content goes into these groups) ---
    with dpg.collapsing_header(label="2. Parameters (Weights & Biases)", default_open=False):
        dpg.add_group(tag="weights_biases_group") # Weights and Biases UI will be built here
    dpg.add_separator()

    with dpg.collapsing_header(label="3. Activation Functions & Training Config", default_open=False):
        dpg.add_group(tag="activations_group") # Activation function combos
        dpg.add_combo(LOSS_FUNCTIONS, label="Loss Function", tag="loss_function_combo", default_value=LOSS_FUNCTIONS[0])
        dpg.add_input_float(label="Learning Rate", tag="learning_rate_input", default_value=network_config["learning_rate"], min_value=0.00001, format="%.5f", width=120)
    dpg.add_separator()

    # --- Input/Output Data ---
    with dpg.collapsing_header(label="4. Data (Inputs & Targets)", default_open=False):
        dpg.add_group(tag="io_values_group") # Input and Target value fields
    dpg.add_separator()

    # --- Operations ---
    with dpg.collapsing_header(label="5. Operations", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Forward Pass", callback=forward_pass_callback)
            dpg.add_button(label="Calculate Loss", callback=calculate_loss_callback)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Backward Pass (Gradients)", callback=backward_pass_callback)
            dpg.add_button(label="Update Parameters", callback=update_parameters_callback)
    dpg.add_separator()

    # --- Results & Gradients Display ---
    with dpg.collapsing_header(label="6. Results (Outputs & Loss)", default_open=True):
        dpg.add_group(tag="results_group") # Calculated outputs and loss text
    dpg.add_separator()
    
    with dpg.collapsing_header(label="7. Calculated Gradients", default_open=False):
        dpg.add_group(tag="gradients_display_group") # Gradient text displays
    dpg.add_separator()

    dpg.add_text("Status: Ready", tag=ID_STATUS_TEXT)

# --- Initialize ---
# Call build once at start to have a default network UI
build_network_structure_and_ui()


dpg.create_viewport(title='NN Step-by-Step Simulator', width=1050, height=850)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window(ID_PRIMARY_WINDOW, True)
dpg.start_dearpygui()
dpg.destroy_context()