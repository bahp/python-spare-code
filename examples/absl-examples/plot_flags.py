
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', '', 'File containing text to pass to the model. Only one of '
    '--input and --input_file can be specified.')
flags.DEFINE_string('attribute_json', '', 'Path to save attribution JSON to.')
flags.DEFINE_string('restore_json', '', 'Path to save restoration JSON to.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.input and not FLAGS.input_file:
    input_text = FLAGS.input
  elif not FLAGS.input and FLAGS.input_file:
    with open(FLAGS.input_file, 'r', encoding='utf8') as f:
      input_text = f.read()
  else:
    raise app.UsageError('Specify exactly one of --input and --input_file.')

  if not 50 <= len(input_text) <= 750:
    raise app.UsageError(
        f'Text should be between 50 and 750 chars long, but the input was '
        f'{len(input_text)} characters')


if __name__ == '__main__':
  app.run(main)