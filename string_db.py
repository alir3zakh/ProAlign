"""
this module contains codes that are responsible to read and parse information
from the string db data source
"""

import utils
import organism
import constants as cs


# parse the string db ppi file
@utils.time_it
def parse_organism_ppi(org, ppi_path, node_path, edge_path):
    with open(ppi_path, 'r') as index:
        # list of all protein codes
        nodes = set()
        # list of all protein interactions
        edges = []

        while True:
            line = index.readline()
            if line == '':
                break
            elif line[:(len(org))] != org:
                continue
            line = line.strip()
            words = line.split()
            prot1 = words[0]
            prot2 = words[1]
            score = words[2]
            if int(score) > cs.INTERACTION_THR:
                nodes.add(prot1)
                nodes.add(prot2)
                edges.append(words)

        utils.write_json(list(nodes), node_path)

        utils.write_json(edges, edge_path)


# parse the string db protein sequences file
@utils.time_it
def parse_organism_seq(org, in_path=cs.STRING_PATH,
                       out_path=cs.JSON_PATH):
    seq_name = '{}.protein.sequences.v10.5.fa'.format(org)
    pseq_name = '{}_parsed_sequences.json'.format(org)

    seq_path = utils.join_path(in_path, seq_name)
    pseq_path = utils.join_path(out_path, pseq_name)

    with open(seq_path, 'r') as index:

        # list of all protein sequences
        proteins = []
        prt = ''
        seq = ''

        while True:
            line = index.readline()
            line = line.strip()
            if line == '':
                proteins.append({'code': prt, 'sequence': seq})
                break
            elif line[0] == '>':
                proteins.append({'code': prt, 'sequence': seq})
                prt = line[1:]
                seq = ''
            else:
                seq += line

        utils.write_json(proteins, pseq_path)


# parse ppi files if need be
def parse_organism(org, in_path=cs.STRING_PATH,
                   out_path=cs.JSON_PATH, check=True):
    ppi_name = '{}.protein.links.v10.5.txt'.format(org)
    node_name = '{}_parsed_nodes.json'.format(org)
    edge_name = '{}_parsed_edges.json'.format(org)

    ppi_path = utils.join_path(in_path, ppi_name)
    node_path = utils.join_path(out_path, node_name)
    edge_path = utils.join_path(out_path, edge_name)

    if (check and utils.files_exist([node_name, edge_name], out_path)):
        message = 'using existing parsed jsons for {}'.format(org)
        utils.print_log(message)

    else:
        message = ('parsing ppi information of {}').format(org)
        utils.print_log(message)

        parse_organism_ppi(org, ppi_path, node_path, edge_path)

        message = ('ppi parsing finished for {}').format(org)
        utils.print_log(message)

    return organism.Organism(nodes_file=node_path,
                             edges_file=edge_path,
                             org_id=org)


# check if initial files are present
def check_initial_files(org, in_path=cs.STRING_PATH):
    # the list of initial files needed for code execution
    check_list = [
        '.protein.links.v10.5.txt',
        '.protein.sequences.v10.5.fa',
    ]

    file_names = [org + x for x in check_list]

    if utils.files_exist(file_names, in_path):
        message = ('initial files check passed for {}').format(org)
        utils.print_log(message)

    else:
        err = ('the initial files for {} not found').format(org)
        utils.print_log(err, mode='err')


# extract single organism from full GO file
def extract_organism_GO(org, go_file=cs.GO_FILE,
                        out_path=cs.JSON_PATH):
    out_file = utils.join_path(out_path, '{}-GO.json'.format(org))

    if utils.file_exists(out_file, ''):
        message = ('GO json already exists for {}').format(org)
        utils.print_log(message)

        return utils.load_json(out_file)

    else:
        with open(go_file, 'r') as index:
            # list of all gene annotations for all proteins
            go_dict = {}

            # start reading GO file
            message = 'Extracting GO information for {}'.format(org)
            utils.print_log(message)

            line_count = 1
            while True:
                if line_count % cs.GO_REPORT_FREQ == 0:
                    message = 'reached line #{}'.format(line_count)
                    utils.print_log(message, mode='progress')
                line_count += 1

                line = index.readline()
                if line == '':
                    break
                elif line.split('\t')[0] != org:
                    continue
                line = line.strip()
                words = line.split('\t')
                prot = words[0]
                go = words[2]
                evidence = words[5]
                score = words[6]
                # if int(score) > INTERACTION_THR:
                go_dict[prot] = go_dict.get(prot, []) + [(go, evidence, score)]

            message = 'Extracting GO for {} finished!'.format(org)
            utils.print_log(message, mode='end_progress')

            utils.write_json(go_dict, out_file)

            return go_dict


# extract single organism from full GO file
def extract_all_organism_GO(go_file, out_path=cs.JSON_PATH):
    in_file = utils.join_path(cs.STRING_PATH, go_file)
    out_file = utils.join_path(out_path, '{}-GO.json'.format(go_file))

    if utils.file_exists(out_file, ''):
        message = ('GO json already exists for {}').format(go_file)
        utils.print_log(message)

        return utils.load_json(out_file)

    else:
        with open(in_file, 'r') as index:
            # list of all gene annotations for all proteins
            go_dict = {}

            # start reading GO file
            message = 'Extracting GO information for {}'.format(go_file)
            utils.print_log(message)

            line_count = 1
            while True:
                if line_count % cs.GO_REPORT_FREQ == 0:
                    message = 'reached line #{}'.format(line_count)
                    utils.print_log(message, mode='progress')
                line_count += 1
                line = index.readline()

                if line == '':
                    break

                line = line.strip()
                words = line.split('\t')
                org = words[0]
                prot = words[1]
                go = words[3]
                evidence = words[6]
                # score = words[7]
                prot_id = '{}.{}'.format(org, prot)
                # if int(score) > INTERACTION_THR:
                if evidence in ['EXP', 'IDA', 'IMP', 'IGI', 'IEP', 'IPI']:
                    go_dict[prot_id] = go_dict.get(prot_id, []) + [go]

            message = 'Extracting GO for {} finished!'.format(go_file)
            utils.print_log(message, mode='end_progress')

            utils.write_json(go_dict, out_file)

            return go_dict


def get_uniprot_map(org, in_path=cs.STRING_PATH):
    file_name = '{}.uniprot.tsv'.format(org)
    with open(utils.join_path(cs.STRING_PATH,
                              file_name), 'r') as mapping:

        # uniprot to string id mapping dict
        uni2sdb = {}

        while True:
            line = mapping.readline()
            line = line.strip()
            if line == '':
                break
            elif line[0] == '#':
                continue
            else:
                words = line.split('\t')
                org_id = words[0]
                uniprot = words[1]
                uniprot_ac, uniprot_id = uniprot.split('|')
                str_id = words[2]
                if org != org_id:
                    raise Exception('org id does not match for mapping!')
                uni2sdb[uniprot_ac] = '{}.{}'.format(org, str_id)

        return uni2sdb
