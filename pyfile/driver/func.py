import configparser
import os
import util

num_fils = 309
gmres_tol = 4
precon = 'precon'

class DRIVER:

    def __init__(self):
        self.globals_name = 'input/globals.ini'
        self.afix = ''
        self.inputfile = f""

        self.category = f'{precon}_wave_N15_gmrestol_{gmres_tol}/'
        self.exe_name = f'cilia_1e-4'
        self.date = '20250310'
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

        self.pars_list = {
                     "index": [],
                     "nswim": [],
                     "nseg": [],
                     "nfil": [],
                     "nblob": [],
                     "ar": [],
                     "spring_factor": [],
                     "tilt_angle": [],
                     "force_mag": [],
                     "seg_sep": [],
                     "period": [],
                     "sim_length": [],
                     "nx": [],
                     "ny": [],
                     "nz": [],
                     "boxsize": [],
                     "fil_spacing": [],
                     "blob_spacing": [],
                     "fil_x_dim": [],
                     "blob_x_dim": [],
                     "hex_num": [],
                     "reverse_fil_direction_ratio": [],
                     "f_eff": [],
                     "theta_0": [],
                     "freq_shift": []}

        # self.sweep_shape = (1, 12, 4, 1)
        self.sweep_shape = (1, 1, 1, 1)

        self.num_sim = 0

        self.current_thread = 0
        self.num_thread = 1
        self.cuda_device = 0
    
    def update_date(self, date):
        self.date = date
        self.dir = f"data/{self.category}{self.date}{self.afix}/"

    def create_ini(self):
        ini = configparser.ConfigParser()
        ini.add_section('Parameters')
        ini.add_section('Filenames')
        ini.add_section('Box')
        ini.add_section('Hex')
        ini.add_section('Concentric')
        ini.add_section('Seeding_util')
        with open(self.globals_name, 'w') as configfile:
            ini.write(configfile, space_around_delimiters=False)
        
    def write_ini(self, section, variable, value):
        ini = configparser.ConfigParser()
        ini.read(self.globals_name)

        ini.set(section, variable, f'{value}')

        # Save the changes back to the file
        with open(self.globals_name, 'w') as configfile:
            ini.write(configfile, space_around_delimiters=False)

    def create_rules(self):
        # Define the rule of sweeping simulations
        index = 0
        for i in range(self.sweep_shape[0]):
            for j in range(self.sweep_shape[1]):
                for k in range(self.sweep_shape[2]):
                    for l in range(self.sweep_shape[3]):

                        seg_sep = 2.6
                        force_mag = 1
                        tilt_angle = 0.2181662   # Platynaereis

                        nfil = num_fils
                        nblob = 9000
                        nseg = 20
                        ar = 8  # This is D/L, not R/L. This is for Platynaereis
                        # Since D/L = 8 and L = 20um, D = 160um
                        # In the simulation, L is around  49 units
                        period = 1
                        spring_factor = 1e-3

                        nx=500
                        ny=500
                        nz=500
                        boxsize=8000
                        fil_spacing=80.0
                        blob_spacing=8.0
                        fil_x_dim=16*(i+1)
                        blob_x_dim=160*(i+1)
                        hex_num=2
                        reverse_fil_direction_ratio=0.0
                        sim_length = 1.0
                        f_eff = 0.3
                        theta_0 = 3.14159265359/2.1
                        freq_shift = 0.0  # This was for a frequency gradient study


                        self.pars_list["index"].append(index)
                        self.pars_list["nswim"].append(1)
                        self.pars_list["nseg"].append(nseg)
                        self.pars_list["nfil"].append(nfil)
                        self.pars_list["nblob"].append(nblob)
                        self.pars_list["ar"].append(ar)
                        self.pars_list["spring_factor"].append(spring_factor)
                        self.pars_list["force_mag"].append(force_mag)
                        self.pars_list["seg_sep"].append(seg_sep)
                        self.pars_list["period"].append(period)
                        self.pars_list["sim_length"].append(sim_length)
                        self.pars_list["tilt_angle"].append(tilt_angle)
                        self.pars_list["nx"].append(nx)
                        self.pars_list["ny"].append(ny)
                        self.pars_list["nz"].append(nz)
                        self.pars_list["boxsize"].append(boxsize)
                        self.pars_list["fil_spacing"].append(fil_spacing)
                        self.pars_list["blob_spacing"].append(blob_spacing)
                        self.pars_list["fil_x_dim"].append(fil_x_dim)
                        self.pars_list["blob_x_dim"].append(blob_x_dim)
                        self.pars_list["hex_num"].append(hex_num)
                        self.pars_list["reverse_fil_direction_ratio"].append(reverse_fil_direction_ratio)
                        self.pars_list["f_eff"].append(f_eff)
                        self.pars_list["theta_0"].append(theta_0)
                        self.pars_list["freq_shift"].append(freq_shift)

                        index += 1
        # Write rules to sim list file
        self.write_rules()

    def delete_files(self):
        util.delete_files_in_directory(self.dir)

    def view_files(self):
        util.view_files_in_directory(self.dir)
        print(f"\033[32m{self.dir}\033[m")
        print(f"\033[34m{self.exe_name}\033[m")

    def write_rules(self):
        os.system(f'mkdir -p {self.dir}')
        sim = configparser.ConfigParser()
        sim.add_section('Parameter list')
        for key, value in self.pars_list.items():
            sim['Parameter list'][key] = ', '.join(map(str, value))
        with open(self.dir+"rules.ini", 'w') as configfile:
            sim.write(configfile, space_around_delimiters=False)

    def read_rules(self):
        sim = configparser.ConfigParser()
        try:
            print("Here")
            sim.read(self.dir+"rules.ini")
            print("Now here")
            for key, value in self.pars_list.items():
                print("Nowwww here")
                if(key in sim["Parameter list"]):
                    print(f"{key}")
                    self.pars_list[key] = [float(x) for x in sim["Parameter list"][key].split(', ')][0::1]
                print("whaaat")
            self.num_sim = len(self.pars_list["nfil"])
            print("what")
        except:
            print("WARNING: " + self.dir + "rules.ini not found.")

    def run(self):
        self.create_ini()
        self.write_ini("Filenames", "simulation_dir", self.dir)

        # Read rules from the sim list file
        self.read_rules()

        thread_list = util.even_list_index(self.num_sim, self.num_thread)
        sim_index_start = thread_list[self.current_thread]
        sim_index_end = thread_list[self.current_thread+1]

        print(f"Partitioning {self.num_sim} into {self.num_thread} threads\n" +\
              f"Partition index: {self.current_thread} / {self.num_thread-1} \n" + \
              f"[{sim_index_start} - {sim_index_end}] / {thread_list}\n" +\
              f"on GPU: {self.cuda_device}")
        
        # Iterate through the sim list and write to .ini file and execute
        for i in range(sim_index_start, sim_index_end):
            
            for key, value in self.pars_list.items():
                self.write_ini("Parameters", key, float(self.pars_list[key][i]))
            self.simName = f"ciliate_{self.pars_list['nfil'][i]:.0f}fil_{self.pars_list['nblob'][i]:.0f}blob_{self.pars_list['ar'][i]:.2f}R_{self.pars_list['spring_factor'][i]:.4f}torsion_{self.pars_list['tilt_angle'][i]:.4f}tilt_{self.pars_list['f_eff'][i]:.4f}f_eff_{self.pars_list['theta_0'][i]:.4f}theta0_{self.pars_list['freq_shift'][i]:.4f}freqshift"
            self.write_ini("Filenames", "simulation_file", self.simName)
            self.write_ini("Filenames", "simulation_dir", self.dir)
            self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d2_N160.dat")
            # self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d3_N640.dat")
            # self.write_ini("Filenames", "filplacement_file_name", f"input/placement/icosahedron/icosa_d4_N2560.dat")
            self.write_ini("Filenames", "blobplacement_file_name", f"input/placement/icosahedron/icosa_d6_N40962.dat")
            # self.write_ini("Filenames", "blobplacement_file_name", f"input/placement/icosahedron/icosa_d4_N2562.dat")
            self.write_ini("Filenames", "simulation_icstate_name", f"{self.dir}psi{i}.dat")
            self.write_ini("Filenames", "simulation_bodystate_name", f"{self.dir}bodystate{i}.dat")
            self.write_ini("Filenames", "cufcm_config_file_name", f"input/simulation_info_cilia")


            

            # command = f"export OPENBLAS_NUM_THREADS=1; \
            #             export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
            #             ./bin/{self.exe_name} > terminal_outputs/output_{self.date}_{self.pars_list['nfil'][i]:.0f}fil_{i}.out"

            command = f"export OPENBLAS_NUM_THREADS=1; \
                        export CUDA_VISIBLE_DEVICES={self.cuda_device}; \
                        nohup ./bin/{self.exe_name} > {precon}_nohup_{gmres_tol}.out &"
                        # ./bin/{self.exe_name}"
            
            # on ic hpc
            # command = f"export OPENBLAS_NUM_THREADS=1; \
            #             ./bin/{self.exe_name}"


            os.system(command)
