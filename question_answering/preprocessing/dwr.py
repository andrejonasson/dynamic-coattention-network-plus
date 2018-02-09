""" CS224n, altered """

import zipfile
import os
import argparse
from squad_preprocess import maybe_download


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_source", choices=['wiki', 'crawl_ci', 'crawl_cs'], required=True)  # added for use with different glove sources
    args = parser.parse_args()

    glove_base_url = "http://nlp.stanford.edu/data/"
    if args.glove_source == 'wiki':
        glove_filename = "glove.6B.zip"
    elif args.glove_source == 'crawl_ci':
        glove_filename = "glove.42B.300d.zip"
    elif args.glove_source == 'crawl_cs':
        glove_filename = "glove.840B.300d.zip"
    
    prefix = os.path.join("download", "dwr")

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if args.glove_source == 'wiki':
        glove_zip = maybe_download(glove_base_url, glove_filename, prefix, 862182613)
    else:
        glove_zip = maybe_download(glove_base_url, glove_filename, prefix)
    glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, glove_filename), 'r')

    glove_zip_ref.extractall(prefix)
    glove_zip_ref.close()
