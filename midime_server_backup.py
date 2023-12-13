import zerorpc

from magenta.models.music_vae import configs as vae_configs

from midiMe import midime_generate as midime
from midiMe import configs

class MidiGenerator(object):

    """Object for ZeroRPC to expose"""

    def __init__(self):
        self.batch_size=8
        self.num_outputs=1
        self.temperature=0.5

        self.config_map=configs.CONFIG_MAP
        self.vae_config_map=vae_configs.CONFIG_MAP

        self.vae_config_name="hierdec-trio_16bar"
        self.config_name="hierdec-trio_16bar"
        self.vae_checkpoint_file="./music_vae_data/pretrained/hierdec-trio_16bar/hierdec-trio_16bar.tar"
        
        self.output_dir_parent="./midi_output/"

        self.last_midi_filename=None
    
    def generate(self,genre,latent_z):
        """Specifies genre to choose from"""
        
        checkpoint_file=f"./music_vae_data/{genre}_trio_16_bar/train/"
        output_dir=self.output_dir_parent+genre

        self.last_midi_filename=midime.run(genre,
                            self.vae_config_name,
                            self.config_name,
                            self.vae_config_map,
                            self.config_map,
                            self.vae_checkpoint_file,
                            checkpoint_file,
                            self.batch_size,
                            self.num_outputs,
                            self.temperature,
                            latent_z,
                            output_dir)

        return
        
    def get_last_midi_filename(self):
        return self.last_midi_filename

    
server=zerorpc.Server(MidiGenerator())
server.bind("tcp://0.0.0.0:4242")
# server.run()

zerorpc.gevent.spawn(server.run)
while True:
    zerorpc.gevent.sleep(10)
