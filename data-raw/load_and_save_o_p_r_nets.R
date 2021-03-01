td <- tempdir()
# "sudo apt-get install python3-venv"
reticulate::virtualenv_create(td, python = "/usr/bin/python3")
reticulate::use_virtualenv(td, required = TRUE)
reticulate::py_install("torch")
torch_py <- reticulate::import("torch")
reticulate::py_run_file("data-raw/load_and_save_o_p_r_nets.py")
reticulate::virtualenv_remove(td)
