#!/usr/bin/env python3

import os
import sys
import argparse

import h5py
import tables

import pandas as pd
import numpy as np


class ArgumentParser:
    def __init__(self):
        self.parser = self.create_argument_parser()

    def parse_input(self, argv):
        """
        Parse command line input.
        :param argv: given arguments
        :return: parsed arguments
        """
        args = self.parser.parse_args(argv)

        return args

    def create_argument_parser(self):
        """
        Method creating an argument parser
        :return: parser
        """
        parser = argparse.ArgumentParser(description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('-i', '--input', type=self.is_readable_dir, required=True,
                            help="Enter the path to a TriTyper file.".format(os.linesep))

        parser.add_argument('-o', '--output', type=self.is_writable_location, required=True,
                            help="The path to write output to.")

        parser.add_argument('-n', '--study_name', type=str,
                            required=True,
                            help="Enter a study name to use.")

        parser.add_argument('-c', '--chunk_size', type=str,
                            required=False, default=25000,
                            help="The maximum number of variants for a .h5 file. (default = 25000)")

        return parser

    @staticmethod
    def is_readable_dir(directory):
        """
        Checks whether the given directory is readable
        :param directory: a path to a directory in string format
        :return: directory
        :raises: Exception: if the given path is invalid
        :raises: Exception: if the given directory is not accessible
        """
        if not os.path.isdir(directory):
            raise Exception("directory: {0} is not a valid directory".format(directory))
        if os.access(directory, os.R_OK):
            return directory
        else:
            raise Exception("directory: {0} is not a readable dir".format(directory))

    def is_writable_location(self, path):
        """
        Checks if the base of the given path represents a location in which writing is permitted.
        :param path: The path to check
        :return: The checked path
        """
        self.is_readable_dir(os.path.dirname(os.path.abspath(path)))
        return path


class TriTyperDataException(Exception):
    pass


class TriTyperData:
    POSSIBLE_ALLELES = ["A", "C", "T", "G", "U", "I", "N"]
    PANDAS_DF_CHUNK_SIZE = 10 ** 6

    def __init__(self, name):
        self.abs_path = os.path.abspath(name)
        self.genotype_matrix_file_path = os.path.join(self.abs_path, "GenotypeMatrix.dat")
        self.dosage_matrix_file_path = os.path.join(self.abs_path, "ImputedDosageMatrix.dat")
        self.snps_file_path = os.path.join(self.abs_path, "SNPs.txt")
        self.snp_mappings_path = os.path.join(self.abs_path, "SNPMappings.txt")
        self.number_of_variants = 0

        print("Loading TriTyper data from '{}'...".format(self.abs_path))

        if not os.path.exists(self.abs_path):
            raise TriTyperDataException("Path '{}' does not exist".format(self.abs_path))

        if not (os.path.exists(self.genotype_matrix_file_path) and os.path.isfile(self.genotype_matrix_file_path)):
            raise TriTyperDataException("Genotype path '{}' not a file".format(self.genotype_matrix_file_path))

        if not (os.path.exists(self.dosage_matrix_file_path) and os.path.isfile(self.dosage_matrix_file_path)):
            raise TriTyperDataException("Dosage matrix path '{}' not a file".format(self.dosage_matrix_file_path))

        # Read the individual data
        self.individuals_data = self.read_individuals_data()
        print("Loaded {} individuals".format(len(self.individuals_data)))
        # Reads the variants in chunks into the variants file reader.
        self._variants_file_reader = self.read_variants()
        print("Loaded {} variants".format(self.number_of_variants))

    def read_individuals_data(self):
        allele_recoding_file_path = os.path.join(self.abs_path, "Individuals.txt")
        try:
            return pd.read_csv(allele_recoding_file_path, header=None,
                               names = ["individual"],
                               dtype={"individual": str})
        except IOError as e:
            raise TriTyperDataException("Could not read '{}'. {}".format(allele_recoding_file_path, str(e)))

    def read_variants(self):
        try:
            # Check the number of snps within the snps file
            with open(self.snps_file_path) as snps_file:
                for i, l in enumerate(snps_file):
                    pass

            self.number_of_variants = i + 1


        except IOError as e:
            raise TriTyperDataException("Could not read '{}'".format(self.snps_file_path), str(e))

        # Load the variants from the snp mappings file
        variants_file_reader = pd.read_csv(self.snp_mappings_path, header=None, sep="\t",
                                           names=["CHR", "bp", "IDS"],
                                           chunksize=self.PANDAS_DF_CHUNK_SIZE)

        return variants_file_reader

    def get_variant_chunks(self):
        # Initialize counter for the number of snps
        number_of_snps_from_mapping_file = 0
        # Loop through the chunks in the variants file reader
        for variant_chunk in self._variants_file_reader:
            # Split the IDS field in ID and a remaining bit
            variant_chunk['ID'] = variant_chunk['IDS'].str.split(",", n=1, expand=True)[0]
            # 'return' a variant chunk with id in every iteration
            yield variant_chunk

            number_of_snps_from_mapping_file += variant_chunk.shape[0]

        # Raise an exception if the number of variants read earlier
        # is not equal to the number encountered here
        if self.number_of_variants != number_of_snps_from_mapping_file:
            raise TriTyperDataException(
                "SNP data from '{}' and '{}' do not have equal lengths ({} vs {} respectively)"
                .format(self.snps_file_path, self.snp_mappings_path,
                        self.number_of_variants, number_of_snps_from_mapping_file))

    def read_genotype_matrix(self, trityper_variant_index):
        try:
            with open(self.genotype_matrix_file_path, "rb") as genotype_matrix_file:
                genotype_matrix_file.seek(trityper_variant_index * (2 * len(self.individuals_data)))
                variant_sample_alleles = list()
                genotype_matrix_bytearray = bytearray(genotype_matrix_file.read(2 * len(self.individuals_data)))

                for i in range(len(self.individuals_data)):
                    allele_index_second = i + len(self.individuals_data)
                    first_allele = chr(genotype_matrix_bytearray[i])
                    second_allele = chr(genotype_matrix_bytearray[allele_index_second])
                    variant_sample_alleles.append(
                        (first_allele if first_allele in self.POSSIBLE_ALLELES else None,
                         second_allele if second_allele in self.POSSIBLE_ALLELES else None))
                return variant_sample_alleles

        except IOError as e:
            raise TriTyperDataException("Could not read '{}'. {}".format(self.genotype_matrix_file_path, str(e)))

    def get_dosages(self, trityper_variant_index):
        dosage_values_float = list()
        dosage_values = list()
        try:
            with open(self.dosage_matrix_file_path, "rb") as dosage_matrix_file, open(self.genotype_matrix_file_path,
                                                                                      "rb") as genotype_matrix_file:

                dosage_matrix_file.seek(trityper_variant_index * len(self.individuals_data))
                dosage_matrix_bytearray = bytearray(dosage_matrix_file.read(len(self.individuals_data)))

                genotype_matrix_file.seek(trityper_variant_index * (2 * len(self.individuals_data)))
                genotype_matrix_bytearray = bytearray(genotype_matrix_file.read(2 * len(self.individuals_data)))

                take_complement = False
                for i in range(len(dosage_matrix_bytearray)):
                    dosage_byte = dosage_matrix_bytearray[i]
                    dosage_values.append(dosage_byte)
                    if dosage_byte != 127:
                        if dosage_byte > 127:
                            dosage_byte = (256 - dosage_byte) * (-1)
                            dosage_values[i] = dosage_byte
                        dosage_value = float(128 + dosage_byte) / 100

                        if genotype_matrix_bytearray[i] == 0 and dosage_value > 1:
                            take_complement = True
                            break
                        if genotype_matrix_bytearray[i] == 2 and dosage_value < 1:
                            take_complement = True
                            break

                if take_complement:
                    for i in range(len(dosage_values)):
                        dosage_byte = dosage_values[i]
                        if dosage_byte != 127:
                            dosage_byte = 200 - (128 + dosage_byte + -128)
                            dosage_values[i] = dosage_byte

                for dosage_byte in dosage_values:
                    if dosage_byte == 127:
                        dosage_values_float.append(-1)
                    else:
                        dosage_values_float.append(float(128 + int(dosage_byte)) / 100)

        except IOError as e:
            raise TriTyperDataException("Could not read '{}'. {}".format(self.dosage_matrix_file_path, str(e)))

        return dosage_values_float

    def get_alleles(self, trityper_variant_index):
        variant_sample_alleles = self.read_genotype_matrix(trityper_variant_index)
        alleles = list()

        for allele_pair in variant_sample_alleles:
            for allele in allele_pair:
                if allele is not None and allele not in alleles:
                    alleles.append(allele)
        return alleles


class HaseHDF5WriterException(Exception):
    pass


class HaseHDF5Writer:
    def __init__(self, path, chunk_size, study_name):
        self.chunk_size = chunk_size
        self.abs_path = os.path.abspath(path)
        self.study_name = study_name
        self.bad_variant_indices = list()

        self.probes_directory_path = os.path.join(self.abs_path, "probes")
        try:
            print("Creating probes folder at {}..."
                  .format(self.probes_directory_path))
            os.makedirs(self.probes_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists"
                                          .format(self.probes_directory_path), e)

        self.individuals_directory_path = os.path.join(self.abs_path, "individuals")
        try:
            print("Creating individuals folder at {}..."
                  .format(self.individuals_directory_path))
            os.mkdir(self.individuals_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists"
                                          .format(self.individuals_directory_path), e)

        self.genotype_directory_path = os.path.join(self.abs_path, "genotype")
        try:
            print("Creating genotype folder at {}..."
                  .format(self.genotype_directory_path))
            os.mkdir(self.genotype_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists"
                                          .format(self.genotype_directory_path), e)

    def write(self, trityper_data):
        self._write_probes(trityper_data)
        print("Completed probes file")
        self._write_individuals(trityper_data)
        print("Completed individuals file")
        self._write_genotype(trityper_data)
        print("Completed genotype file")

    def _write_probes(self, trityper_data):
        probes_path = os.path.join(self.probes_directory_path, self.study_name + '.h5')
        if os.path.isfile(probes_path):
            raise HaseHDF5WriterException("File '{}' already exists".format(probes_path))

        hash_table = {'keys': np.array([], dtype=np.int), 'allele': np.array([])}

        variant_index = 0
        for i, variant_chunk in enumerate(trityper_data.get_variant_chunks()):

            alleles1 = list()
            alleles2 = list()
            chunked_variant_index = 0
            variant_chunk_length = len(variant_chunk)
            bad_variant_indices_probes_chunk = list()
            while chunked_variant_index < variant_chunk_length:
                alleles = trityper_data.get_alleles(variant_index)
                if len(alleles) != 2:
                    print("Not found 2 alleles for variant '{}': discarding variant..."
                          .format(variant_chunk["ID"][chunked_variant_index]), file=sys.stderr)
                    print(variant_index, chunked_variant_index)
                    self.bad_variant_indices.append(variant_index)
                    bad_variant_indices_probes_chunk.append(chunked_variant_index)

                    # if len(alleles) == 1:
                    #     print("len alleles is 1", file=sys.stderr)
                    #     alleles.append(alleles[0])
                    # elif len(alleles) == 0:
                    #     print("len alleles is 0", file=sys.stderr)
                    #     alleles = ["N", "N"]
                else:
                    alleles1.append(alleles[0])
                    alleles2.append(alleles[1])

                variant_index += 1
                chunked_variant_index += 1

            # Drop every variant that does not have exactly 2 alleles
            filtered_variant_chunk = variant_chunk.drop(
                bad_variant_indices_probes_chunk)

            filtered_variant_chunk = filtered_variant_chunk.reset_index(drop=True)

            alleles1_series = pd.Series(alleles1)
            alleles2_series = pd.Series(alleles2)

            hash_1 = alleles1_series.apply(hash)
            hash_2 = alleles2_series.apply(hash)

            k, indices = np.unique(np.append(hash_1, hash_2), return_index=True)
            s = np.append(alleles1_series, alleles2_series)[indices]
            ind = np.invert(np.in1d(k, hash_table['keys']))
            hash_table['keys'] = np.append(hash_table['keys'], k[ind])
            hash_table['allele'] = np.append(hash_table['allele'], s[ind])
            filtered_variant_chunk["allele1"] = hash_1
            filtered_variant_chunk["allele2"] = hash_2

            filtered_variant_chunk.to_hdf(probes_path,
                           data_columns=["CHR", "bp", "ID", 'allele1', 'allele2'],
                           key='probes', format='table', append=True,
                           min_itemsize=25, complib='zlib', complevel=9, dropna=True)

            print("Wrote {} variants to probes file".format(variant_index))

        pd.DataFrame.from_dict(hash_table).to_csv(
            os.path.join(self.probes_directory_path, self.study_name + '_hash_table.csv.gz'),
            index=False, compression='gzip', sep='\t')

    def _write_individuals(self, trityper_data):
        individuals_path = os.path.join(self.individuals_directory_path, self.study_name + '.h5')
        if os.path.isfile(individuals_path):
            raise HaseHDF5WriterException("File '{}' already exists".format(individuals_path))

        trityper_data.individuals_data.to_hdf(
            individuals_path, key='individuals', format='table',
            min_itemsize=25, complib='zlib',complevel=9)

    def _write_genotype(self, trityper_data):

        number_of_chunks = (trityper_data.number_of_variants // self.chunk_size) + 1
        for chunk_index in range(number_of_chunks):
            start = chunk_index * self.chunk_size
            end = min((chunk_index + 1) * self.chunk_size, trityper_data.number_of_variants)
            dosage_matrix = np.empty((end-start, len(trityper_data.individuals_data)))

            print("Loading {}-{} variants to write to chunk {} out of {} total chunks".format(
                start, end, chunk_index, number_of_chunks))

            # Get the dosages for every variant within the
            for chunked_variant_index, variant_index in enumerate(range(start, end)):
                dosage_matrix[chunked_variant_index, ] = trityper_data.get_dosages(variant_index)

            # Drop every variant that did not have two alleles.
            bad_variant_indices_chunk = list()
            for bad_variant_index in self.bad_variant_indices:
                if start <= bad_variant_index < end:
                    bad_variant_indices_chunk.append(bad_variant_index - start)
            print(bad_variant_indices_chunk)
            dosage_matrix = np.delete(dosage_matrix, bad_variant_indices_chunk, 0)

            h5_gen_file = tables.open_file(
                os.path.join(self.genotype_directory_path, str(chunk_index) + '_' + self.study_name + '.h5'), 'w', title=self.study_name)

            atom = tables.Float16Atom()
            genotype = h5_gen_file.create_carray(h5_gen_file.root, 'genotype', atom,
                                                 (dosage_matrix.shape),
                                                 title='Genotype',
                                                 filters=tables.Filters(complevel=9, complib='zlib'))

            genotype[:] = dosage_matrix
            h5_gen_file.close()
        print("Discarded {} variants that did not have two alleles".format(
            len(self.bad_variant_indices)), file=sys.stderr)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    arguments = ArgumentParser().parse_input(argv)

    # Prepare writer. This will also check if the output path is available
    writer = HaseHDF5Writer(arguments.output, arguments.chunk_size, arguments.study_name)
    # Load the TriTyperData
    trityper_data = TriTyperData(arguments.input)
    # Write the TriTyperData
    writer.write(trityper_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
