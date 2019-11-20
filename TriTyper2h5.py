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

        parser.add_argument('-o', '--output_path', type=self.is_writable_location, required=True,
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

    def __init__(self, name):
        self.abs_path = os.path.abspath(name)
        self.genotype_matrix_file_path = os.path.join(self.abs_path, "GenotypeMatrix.dat")
        self.dosage_matrix_file_path = os.path.join(self.abs_path, "ImputedDosageMatrix.dat")

        print("Loading TriTyper data from '{}'...".format(self.abs_path))

        if not os.path.exists(self.abs_path):
            raise TriTyperDataException("Path '{}' does not exist".format(self.abs_path))

        if not (os.path.exists(self.genotype_matrix_file_path) and os.path.isfile(self.genotype_matrix_file_path)):
            raise TriTyperDataException("Genotype path '{}' not a file".format(self.genotype_matrix_file_path))

        if not (os.path.exists(self.dosage_matrix_file_path) and os.path.isfile(self.dosage_matrix_file_path)):
            raise TriTyperDataException("Dosage matrix path '{}' not a file".format(self.dosage_matrix_file_path))

        self.individuals_data = self.read_individuals_data()
        print("Loaded {} individuals".format(len(self.individuals_data)))
        self.variants = self.read_variants()
        print("Loaded {} variants".format(len(self.variants)))

    def read_individuals_data(self):
        allele_recoding_file_path = os.path.join(self.abs_path, "Individuals.txt")
        try:
            return pd.read_csv(allele_recoding_file_path, header=None, names = ["individuals"])
        except IOError as e:
            raise TriTyperDataException("Could not read '{}'. {}".format(allele_recoding_file_path, str(e)))

    def read_variants(self):
        snps_file_path = os.path.join(self.abs_path, "SNPs.txt")
        snp_mappings_path = os.path.join(self.abs_path, "SNPMappings.txt")

        try:
            snps_data = pd.read_csv(snps_file_path, header=None, names=["ID", "ALTID"])
            snp_mappings_data = pd.read_csv(snp_mappings_path, header=None, sep="\t", names=["CHR", "bp", "IDS"])
            if len(snps_data) != len(snp_mappings_data):
                raise TriTyperDataException("SNP data from '{}' and '{}' do not have equal lengths"
                                            .format(snps_file_path, snp_mappings_path))

            variants = pd.concat([snps_data, snp_mappings_data], axis=1)
            variants["allele1"] = 0
            variants["allele2"] = 0

            return variants
        except IOError as e:
            raise TriTyperDataException("Could not read '{}'. {}".format(snps_file_path, str(e)))

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

        if os.path.isdir(self.abs_path):
            raise HaseHDF5WriterException("Directory '{}' already exists".format(self.abs_path))

        self.probes_directory_path = os.path.join(self.abs_path, "probes")
        try:
            print("Creating probes folder at {}...".format(self.probes_directory_path))
            os.makedirs(self.probes_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists".format(self.probes_directory_path), e)

        self.individuals_directory_path = os.path.join(self.abs_path, "individuals")
        try:
            print("Creating individuals folder at {}...".format(self.individuals_directory_path))
            os.mkdir(self.individuals_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists".format(self.individuals_directory_path), e)

        self.genotype_directory_path = os.path.join(self.abs_path, "genotype")
        try:
            print("Creating genotype folder at {}...".format(self.genotype_directory_path))
            os.mkdir(self.genotype_directory_path)
        except FileExistsError as e:
            raise HaseHDF5WriterException("Directory '{}' already exists".format(self.genotype_directory_path), e)

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

        for variant_index in range(len(trityper_data.variants)):
            alleles = trityper_data.get_alleles(variant_index)
            hash_1 = hash(alleles[0])
            hash_2 = hash(alleles[1])
            k, indices = np.unique(np.append(hash_1, hash_2), return_index=True)
            s = np.append(alleles[0], alleles[1])[indices]
            ind = np.invert(np.in1d(k, hash_table['keys']))
            hash_table['keys'] = np.append(hash_table['keys'], k[ind])
            hash_table['allele'] = np.append(hash_table['allele'], s[ind])

            trityper_data.variants.loc[variant_index, "allele1"] = hash_1
            trityper_data.variants.loc[variant_index, "allele2"] = hash_2
            variant_df = trityper_data.variants.iloc[[variant_index]]

            variant_df.to_hdf(probes_path,
                           data_columns=["CHR", "bp", "ID", 'allele1', 'allele2'],
                           key='probes', format='table', append=True,
                           min_itemsize=25, complib='zlib', complevel=9)

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

        number_of_chunks = (len(trityper_data.variants) // self.chunk_size) + 1
        for chunk_index in range(number_of_chunks):
            start = chunk_index * self.chunk_size
            end = min((chunk_index + 1) * self.chunk_size, len(trityper_data.variants))
            dosage_matrix = np.empty((end-start, len(trityper_data.individuals_data)))

            print("Writing {}-{} variants to h5 chunk {} out of {} total chunks".format(
                start, end, chunk_index, number_of_chunks))

            for variant_index in range(start, end):
                dosage_matrix[variant_index, ] = trityper_data.get_dosages(variant_index)

            h5_gen_file = tables.open_file(
                os.path.join(self.genotype_directory_path, str(chunk_index) + '_' + self.study_name + '.h5'), 'w', title=self.study_name)

            atom = tables.Float16Atom()
            genotype = h5_gen_file.create_carray(h5_gen_file.root, 'genotype', atom,
                                                 (dosage_matrix.shape),
                                                 title='Genotype',
                                                 filters=tables.Filters(complevel=9, complib='zlib'))

            genotype[:] = dosage_matrix
            h5_gen_file.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    arguments = ArgumentParser().parse_input(argv[1:])

    trityper_data = TriTyperData(arguments.input)
    writer = HaseHDF5Writer(arguments.output_path, arguments.chunk_size, arguments.study_name)
    writer.write(trityper_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
