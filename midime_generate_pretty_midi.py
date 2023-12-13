# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modification copyright 2020 Bui Quoc Bao.
# Add Latent Constraint VAE model.
# Add Small VAE model.

"""MidiMe generation script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from magenta import music as mm
from magenta.models.music_vae import configs as vae_configs
import tensorflow.compat.v1 as tf   # pylint: disable=import-error

from . import configs
from .trained_model import TrainedModel


flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.'
)
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.'
)
flags.DEFINE_string(
    'vae_checkpoint_file', None,
    'Path to the MusicVAE checkpoint file.'
)
flags.DEFINE_string(
    'output_dir', 'tmp/generated',
    'The directory where MIDI files will be saved to.'
)
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.'
)
flags.DEFINE_string(
    'vae_config', None,
    'The name of pretrained MusicVAE model'
)
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` model, the number of samples to produce. In `interpolate` '
    'model, the number of steps (including the endpoints).'
)
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.'
)
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.'
)
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.'
)


def run(genre,
        vae_config_name,
        config_name,
        vae_config_map,
        config_map,
        vae_checkpoint_file,
        checkpoint_file,
        batch_size,
        num_outputs,
        temperature,
        output_dir):
    
    """
    Load model params, save config file and start trainer.
    :param config_map: MidiMe dictionary mapping configuration name to Config object.
    :param vae_config_map: MusicVAE dictionary mapping configuration name to Config object.
    :raises:
        ValueError: if required flags are missing or invalid.
    """
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

    if checkpoint_file is None:
        raise ValueError(
            '`--checkpoint_file` must be specified.'
        )
    if vae_checkpoint_file is None:
        raise ValueError(
            '`--vae_checkpoint_file` is required.'
        )
    if output_dir is None:
        raise ValueError('`--output_dir` is required.')
    
    tf.gfile.MakeDirs(output_dir)

    if config_name not in config_map:
        raise ValueError('Invalid MidiMe config name: %s' % config_name)
    config = config_map[config_name]
    
    if vae_config_name not in vae_config_map:
        raise ValueError('Invalid MusicVAE config name: %s' % vae_config_name)
    vae_config = vae_config_map[vae_config_name]
    config.data_converter.max_tensors_per_item = None

    checkpoint_dir_or_path = os.path.expanduser(checkpoint_file)
    vae_checkpoint_dir_or_path = os.path.expanduser(vae_checkpoint_file)

    model = TrainedModel(
        vae_config=vae_config, model_config=config,
        batch_size=min(batch_size, num_outputs),
        vae_checkpoint_dir_or_path=vae_checkpoint_dir_or_path,
        model_checkpoint_dir_or_path=checkpoint_dir_or_path,
        model_var_pattern=['latent'], session_target='')

    logging.info('Sampling...')
    result = model.sample(
        n=num_outputs,
        length=config.hparams.max_seq_len,
        temperature=temperature)[0]
    
    # save_dir=os.path.abspath(os.path.join(output_dir,date_and_time)+".mid")
        
    midi=mm.sequence_proto_to_pretty_midi(result)

    logging.info('Done.')

    return midi

def main(unused_argv):
    """Call generation function."""
    logging.set_verbosity(FLAGS.log)
    run(configs.CONFIG_MAP, vae_configs.CONFIG_MAP)


def console_entry_point():
    """Run entry point."""
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
