import tensorflow_datasets as tfds
from apache_beam.options.pipeline_options import PipelineOptions


data_dir = './data'
local_croissant_file = './scripts/notebooks/schema/f16-croissant.json'
if __name__ == "__main__":

    builder = tfds.core.dataset_builders.CroissantBuilder(
        jsonld=local_croissant_file,
        record_set_ids=['force-to-acceleration'],
        file_format='array_record',
        data_dir=data_dir
    )
    print(builder.info)


    beam_options = PipelineOptions(
        flags = []
    )
    download_config = tfds.download.DownloadConfig(beam_options=beam_options)
    builder.download_and_prepare(download_config=download_config)