"""
this module contains codes that are responsible to communicate with other
existing source codes. (such as blast, netal, isorank, etc)
"""

import os
import numpy as np
from Bio.Blast import NCBIXML

import utils
import string_db
import constants as cs


# blast functions
def create_blast_db(organism, check_path=cs.BLAST_PATH, check=True):
    # the list of file extensions created by makeblastdb
    check_list = [
        '.db.phr',
        '.db.pin',
        '.db.pog',
        '.db.psi',
        '.db.psq',
    ]

    file_names = [organism + x for x in check_list]

    if (check and utils.files_exist(file_names, check_path)):
        message = 'using existing blast db for {}'.format(organism)
        utils.print_log(message)

    else:
        message = ('using makeblastdb to create db for {}').format(organism)
        utils.print_log(message)

        infile = utils.join_path(cs.STRING_PATH,
                                 '{}.protein.sequences.v10.5.fa'.
                                 format(organism))

        outfile = utils.join_path(cs.BLAST_PATH, '{}.db'.format(organism))

        utils.run_cmd(('makeblastdb -in {} '
                       '-parse_seqids -dbtype prot -out {}'
                       ).format(infile, outfile))

        message = ('create db command finished for {}').format(organism)
        utils.print_log(message)


def run_blast_prot(org1, org2, check_path=cs.BLAST_PATH,
                   check=True, eValue='0.01'):
    file_name = '{}-{}.xml'.format(org1, org2)

    if (check and utils.file_exists(file_name, check_path)):
        message = 'using existing blastp results for {}-{}'.format(org1, org2)
        utils.print_log(message)

    else:
        message = 'running blastp query for {}-{}'.format(org1, org2)
        utils.print_log(message)

        file1 = utils.join_path(cs.STRING_PATH,
                                '{}.protein.sequences.v10.5.fa'.
                                format(org1))

        file2 = utils.join_path(cs.BLAST_PATH, '{}.db'.format(org2))

        outfile = utils.join_path(cs.BLAST_PATH, '{}-{}.xml'.
                                                 format(org1, org2))

        utils.run_cmd(('blastp -query {} '
                       '-db {} -out {} -evalue {} -outfmt 5'
                       ).format(file1, file2, outfile, eValue))

        message = 'blastp generated ppi for {}-{}'.format(org1, org2)
        utils.print_log(message)


@utils.time_it
def blast_xml_to_matrix(bio_net, file_path=cs.BLAST_PATH):
    # blast_mat = sparse.lil_matrix(bio_net.dim_sim)
    blast_mat = np.zeros(bio_net.dim_sim)

    result_file = '{}-{}.xml'.format(bio_net.org1.org_id, bio_net.org2.org_id)
    result_file = utils.join_path(file_path, result_file)

    with open(result_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        for blast_record in blast_records:
            s1 = blast_record.query
            for alignment in blast_record.alignments:
                s2 = alignment.title.split()[0]
                # there might be new nodes not connected to anything
                id1 = bio_net.org1.node_to_id.get(s1, None)
                id2 = bio_net.org2.node_to_id.get(s2, None)
                if id1 and id2:
                    blast_mat[bio_net.v_ind(id1, id2)] = alignment.hsps[0].bits

    return blast_mat


@utils.time_it
def self_blast_xml_to_vec(organism, file_path=cs.BLAST_PATH):
    # blast_mat = sparse.lil_matrix(bio_net.dim_sim)
    blast_vec = np.zeros(organism.node_count)

    result_file = '{}-{}.xml'.format(organism.org_id, organism.org_id)
    result_file = utils.join_path(file_path, result_file)

    with open(result_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        for blast_record in blast_records:
            s1 = blast_record.query
            for alignment in blast_record.alignments:
                s2 = alignment.title.split()[0]
                # there might be new nodes not connected to anything
                id1 = organism.node_to_id.get(s1, None)
                if id1 and s1 == s2:
                    blast_vec[id1] = alignment.hsps[0].bits

    return blast_vec


# isorankN functions
def blast_xml_to_eval(org1, org2, file_path=cs.BLAST_PATH,
                      isoN_path=cs.ISON_PATH):
    result_file = '{}-{}.xml'.format(org1.org_id, org2.org_id)
    result_file = utils.join_path(file_path, result_file)

    eval_name = '{}-{}.evals'.format(org1.org_id, org2.org_id)
    eval_file = utils.join_path(isoN_path, eval_name)

    if not os.path.exists(eval_file):
        with open(result_file) as result_handle:
            blast_records = NCBIXML.parse(result_handle)
            with open(eval_file, 'w') as eval_handle:
                for blast_record in blast_records:
                    s1 = blast_record.query
                    for alignment in blast_record.alignments:
                        s2 = alignment.title.split()[0]
                        # there might be new nodes not connected to anything
                        id1 = org1.node_to_id.get(s1, None)
                        id2 = org2.node_to_id.get(s2, None)
                        if id1 and id2:
                            e_score = alignment.hsps[0].bits
                            line = (s1 + '\t' + s2 + '\t' + str(e_score) + '\n')
                            eval_handle.write(line)


def generate_tab_file(organism, file_path=cs.ISON_PATH):
    file_name = '{}.tab'.format(organism.org_id)

    if not os.path.exists(utils.join_path(file_path, file_name)):
        with open(utils.join_path(file_path, file_name), 'w') as tabfile:
            tabfile.write('INTERACTOR_A\tINTERACTOR_B\n')
            for edge in organism.edges:
                es = organism.id_to_node[edge[0]]
                et = organism.id_to_node[edge[1]]
                tabfile.write(es + '\t' + et + '\n')


@utils.time_it
def isorankN_run(bio_net, cluster_file, file_path=cs.ISON_PATH):
    # generate input info
    generate_tab_file(bio_net.org1)
    generate_tab_file(bio_net.org2)

    # generate eval files (blast)
    blast_xml_to_eval(bio_net.org1, bio_net.org1)
    blast_xml_to_eval(bio_net.org1, bio_net.org2)
    blast_xml_to_eval(bio_net.org2, bio_net.org2)

    # generate exec info
    data_input = 'data_{}-{}.inp'.format(
                bio_net.org1.org_id, bio_net.org2.org_id)
    data_file = utils.join_path(file_path, data_input)
    if not os.path.exists(data_file):
        with open(data_file, 'w') as conf:
            conf.write('{}\n'.format(file_path))
            conf.write('-\n')
            conf.write('2\n')
            conf.write(bio_net.org1.org_id + '\n')
            conf.write(bio_net.org2.org_id + '\n')

    # exec_path = utils.join_path(cs.ISON_PATH, 'isorank-n-v3-64')
    # using new version with threading (much faster)
    exec_path = utils.join_path(cs.ISON_PATH, 'isorank-n-s-v3-64')

    cluster_file = os.path.join(file_path, cluster_file)
    exec_command = ('{} --alpha {} '
                    '--o {} --K {} '
                    '--prefix {}/iso --thresh {} '
                    '--maxveclen {} {}').format(exec_path,
                                                cs.ISON_ALPHA,
                                                cluster_file,
                                                cs.ISON_ITERS,
                                                cs.ISON_PATH,
                                                cs.ISON_THRESH,
                                                cs.ISON_MAXVL,
                                                data_file)


    utils.run_cmd(exec_command)


def isorankN_align(bio_net, file_path=cs.ISON_PATH, check=True):
    file_name = 'isorankN_cluster_{}-{}.txt'.format(
                bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing isorankN results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running isorankN for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        isorankN_run(bio_net, file_name)

        message = 'isorankN finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.ISON_PATH, file_name), 'rU') as isoN:
        for line in isoN:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(bio_net.org1.org_id)]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(bio_net.org2.org_id)]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])


                if pair is not None:
                    pairs.append(pair)

    return pairs


# NETAL functions
def blast_xml_to_val(org1, org2, file_path=cs.BLAST_PATH,
                     out_path=cs.NETAL_PATH):
    result_file = '{}-{}.xml'.format(min(org1.org_id, org2.org_id),
                                     max(org1.org_id, org2.org_id))
    result_file = utils.join_path(file_path, result_file)

    eval_name = '{}-{}.val'.format(org1.org_id, org2.org_id)
    eval_file = utils.join_path(out_path, eval_name)

    reverse = ((org1.org_id, org2.org_id) != (min(org1.org_id, org2.org_id),
                                              max(org1.org_id, org2.org_id)))
    with open(result_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        with open(eval_file, 'w') as eval_handle:
            for blast_record in blast_records:
                if not reverse:
                    s1 = blast_record.query
                else:
                    s2 = blast_record.query
                for alignment in blast_record.alignments:
                    if not reverse:
                        s2 = alignment.title.split()[0]
                    else:
                        s1 = alignment.title.split()[0]
                    # there might be new nodes not connected to anything
                    id1 = org1.node_to_id.get(s1, None)
                    id2 = org2.node_to_id.get(s2, None)
                    if id1 and id2:
                        e_score = alignment.hsps[0].bits
                        if not reverse:
                            line = (s1 + '\t' + s2 + '\t' +
                                    str(e_score) + '\n')
                        else:
                            line = (s2 + '\t' + s1 + '\t' +
                                    str(e_score) + '\n')
                        eval_handle.write(line)


def generate_netal_tab_file(organism, file_path=cs.NETAL_PATH):
    file_name = '{}.tab'.format(organism.org_id)
    with open(utils.join_path(file_path, file_name), 'w') as tabfile:
        for edge in organism.edges:
            es = organism.id_to_node[edge[0]]
            et = organism.id_to_node[edge[1]]
            tabfile.write(es + '\t' + et + '\n')


@utils.time_it
def NETAL_run(bio_net, file_path=cs.NETAL_PATH):
    # generate input info
    generate_netal_tab_file(bio_net.org1)
    generate_netal_tab_file(bio_net.org2)

    # generate eval files (blast)
    blast_xml_to_val(bio_net.org1, bio_net.org1)
    blast_xml_to_val(bio_net.org1, bio_net.org2)
    blast_xml_to_val(bio_net.org2, bio_net.org2)

    exec_command = ('./NETAL {}.tab {}.tab -a {} -b {} -c {} '
                    '-i {}').format(bio_net.org1.org_id,
                                    bio_net.org2.org_id,
                                    cs.NETAL_AA,
                                    cs.NETAL_BB,
                                    cs.NETAL_CC,
                                    cs.NETAL_IT)

    utils.run_cmd(exec_command, cwd=cs.NETAL_PATH)


def NETAL_align(bio_net, file_path=cs.NETAL_PATH, check=True):
    file_name = ('({}.tab-{}.tab)-a{}-b{}-c{}-i{}.'
                 'alignment').format(bio_net.org1.org_id,
                                     bio_net.org2.org_id,
                                     cs.NETAL_AA,
                                     cs.NETAL_BB,
                                     cs.NETAL_CC,
                                     cs.NETAL_IT)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing NETAL results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running NETAL for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        NETAL_run(bio_net)

        message = 'NETAL finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.NETAL_PATH, file_name), 'rU') as NETAL:
        for line in NETAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs


# pinalog functions
def blast_xml_to_score(org1, org2, file_path=cs.BLAST_PATH,
                       out_path=cs.PINALOG_PATH):
    result_file = '{}-{}.xml'.format(org1.org_id, org2.org_id)
    result_file = utils.join_path(file_path, result_file)

    eval_name = '{}-{}.blast_score'.format(org1.org_id, org2.org_id)
    eval_file = utils.join_path(out_path, eval_name)

    if not os.path.exists(eval_file):
        with open(result_file) as result_handle:
            blast_records = NCBIXML.parse(result_handle)
            with open(eval_file, 'w') as eval_handle:
                for blast_record in blast_records:
                    s1 = blast_record.query
                    for alignment in blast_record.alignments:
                        s2 = alignment.title.split()[0]
                        # there might be new nodes not connected to anything
                        id1 = org1.node_to_id.get(s1, None)
                        id2 = org2.node_to_id.get(s2, None)
                        if id1 and id2:
                            e_score = alignment.hsps[0].bits
                            line = (s1 + '\t' + s2 + '\t' + str(e_score) + '\n')
                            eval_handle.write(line)

    return eval_name


def generate_pin_file(organism, file_path=cs.PINALOG_PATH):
    file_name = '{}.pin'.format(organism.org_id)

    if not os.path.exists(utils.join_path(file_path, file_name)):
        with open(utils.join_path(file_path, file_name), 'w') as tabfile:
            for edge in organism.edges:
                es = organism.id_to_node[edge[0]]
                et = organism.id_to_node[edge[1]]
                tabfile.write(es + '\t' + et + '\n')


@utils.time_it
def pinalog_run(bio_net, file_path=cs.PINALOG_PATH):
    # generate input info
    generate_pin_file(bio_net.org1)
    generate_pin_file(bio_net.org2)

    # generate eval files (blast)
    score_file = blast_xml_to_score(bio_net.org1, bio_net.org2)

    exec_command = ('./pinalog1.0 {}.pin {}.pin '
                    '{}').format(bio_net.org1.org_id,
                                 bio_net.org2.org_id,
                                 score_file)

    utils.run_cmd(exec_command, cwd=cs.PINALOG_PATH)


def pinalog_align(bio_net, file_path=cs.PINALOG_PATH, check=True):
    file_name = 'net1_net2.pinalog.nodes_algn.txt'

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing pinalog results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running pinalog for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        pinalog_run(bio_net)

        message = 'pinalog finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.PINALOG_PATH, file_name), 'rU') as pinalog:
        for line in pinalog:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])


                if pair is not None:
                    pairs.append(pair)

    return pairs


# C-GRAAL functions
def blast_xml_to_sim(org1, org2, file_path=cs.BLAST_PATH,
                     out_path=cs.CGRAAL_PATH):
    result_file = '{}-{}.xml'.format(org1.org_id, org2.org_id)
    result_file = utils.join_path(file_path, result_file)

    sim_name = '{}-{}-sim.txt'.format(org1.org_id, org2.org_id)
    sim_file = utils.join_path(out_path, sim_name)

    with open(result_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        with open(sim_file, 'w') as sim_handle:
            for blast_record in blast_records:
                s1 = blast_record.query
                for alignment in blast_record.alignments:
                    s2 = alignment.title.split()[0]
                    # there might be new nodes not connected to anything
                    id1 = org1.node_to_id.get(s1, None)
                    id2 = org2.node_to_id.get(s2, None)
                    if id1 and id2:
                        e_score = alignment.hsps[0].bits
                        line = (s1 + '\t' + s2 + '\t' + str(e_score) + '\n')
                        sim_handle.write(line)

    return sim_name


def generate_gw_file(organism, file_path=cs.CGRAAL_PATH):
    file_name = '{}.txt'.format(organism.org_id)

    if not os.path.exists(utils.join_path(file_path, file_name)):
        with open(utils.join_path(file_path, file_name), 'w') as tabfile:
            for edge in organism.edges:
                es = organism.id_to_node[edge[0]]
                et = organism.id_to_node[edge[1]]
                tabfile.write(es + '\t' + et + '\n')

    # exec_command = './list2leda {} >> {}.gw'.format(file_name, organism.org_id)
    exec_command = './list2leda {}'.format(file_name)
    gw_file = '{}.gw'.format(organism.org_id)

    if not os.path.exists(utils.join_path(file_path, gw_file)):
        utils.write_bytes(utils.run_cmd(exec_command, cwd=file_path)[0],
                        utils.join_path(file_path, gw_file))


@utils.time_it
def CGRAAL_run(bio_net, file_path=cs.CGRAAL_PATH):
    # generate input info
    generate_gw_file(bio_net.org1)
    generate_gw_file(bio_net.org2)

    file_names = [
        '{}_{}-CGRAAL.names'.format(
            bio_net.org1.org_id, bio_net.org2.org_id),
        '{}_{}-CGRAAL.nums'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
    ]

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    # generate eval files (blast)
    score_file = blast_xml_to_sim(organism1, organism2)

    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1



    exec_command = ('./CGRAAL_unix64 {}.gw {}.gw '
                    '{} {} {}').format(organism1.org_id,
                                       organism2.org_id,
                                       score_file, file_names[1],
                                       file_names[0])

    utils.run_cmd(exec_command, cwd=cs.CGRAAL_PATH)


def CGRAAL_align(bio_net, file_path=cs.CGRAAL_PATH, check=True):
    file_names = [
        '{}_{}-CGRAAL.names'.format(
            bio_net.org1.org_id, bio_net.org2.org_id),
        '{}_{}-CGRAAL.nums'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
    ]

    if (check and utils.files_exist(file_names, file_path)):
        message = 'using existing CGRAAL results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running CGRAAL for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        CGRAAL_run(bio_net)

        message = 'CGRAAL finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.CGRAAL_PATH, file_names[0]), 'rU') as CGRAAL:
        for line in CGRAAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs


# GRAAL functions
@utils.time_it
def GRAAL_run(bio_net, file_path=cs.GRAAL_PATH):
    # generate input info
    generate_gw_file(bio_net.org1, file_path)
    generate_gw_file(bio_net.org2, file_path)

    # file_name = '{}_{}-GRAAL.aln'.format(
    #     bio_net.org1.org_id, bio_net.org2.org_id)

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    exec_command = ('./GRAALRunner.py 0.8 {}.gw {}.gw '
                    '{}_{}-GRAAL').format(organism1.org_id,
                                          organism2.org_id,
                                          bio_net.org1.org_id,
                                          bio_net.org2.org_id)

    utils.run_cmd(exec_command, cwd=cs.GRAAL_PATH)


def GRAAL_align(bio_net, file_path=cs.GRAAL_PATH, check=True):
    file_name = '{}_{}-GRAAL.aln'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing GRAAL results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running GRAAL for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        GRAAL_run(bio_net)

        message = 'GRAAL finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.GRAAL_PATH, file_name), 'rU') as GRAAL:
        for line in GRAAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])


                if pair is not None:
                    pairs.append(pair)

    return pairs


# MI-GRAAL functions
@utils.time_it
def MIGRAAL_run(bio_net, file_path=cs.MIGRAAL_PATH):
    # generate input info
    generate_gw_file(bio_net.org1, file_path)
    generate_gw_file(bio_net.org2, file_path)

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    exec_command = ('./MI-GRAALRunner.py {}.gw {}.gw '
                    '{}_{}-MI-GRAAL -p 3').format(organism1.org_id,
                                                  organism2.org_id,
                                                  bio_net.org1.org_id,
                                                  bio_net.org2.org_id)

    utils.run_cmd(exec_command, cwd=cs.MIGRAAL_PATH)


def MIGRAAL_align(bio_net, file_path=cs.MIGRAAL_PATH, check=True):
    file_name = '{}_{}-MI-GRAAL.aln'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing MI-GRAAL results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running MI-GRAAL for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        MIGRAAL_run(bio_net)

        message = 'MI-GRAAL finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.MIGRAAL_PATH, file_name), 'rU') as MIGRAAL:
        for line in MIGRAAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])


                if pair is not None:
                    pairs.append(pair)

    return pairs


# HubAlign functions
@utils.time_it
def HubAlign_run(bio_net, file_path=cs.HUBALIGN_PATH):
    # generate input info
    generate_netal_tab_file(bio_net.org1, file_path=cs.HUBALIGN_PATH)
    generate_netal_tab_file(bio_net.org2, file_path=cs.HUBALIGN_PATH)

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    # generate eval files (blast)
    # blast_xml_to_val(bio_net.org1, bio_net.org2, out_path=cs.HUBALIGN_PATH)
    blast_xml_to_val(organism1, organism2, out_path=cs.HUBALIGN_PATH)

    exec_command = ('./HubAlign {}.tab {}.tab -l 0.1 -a 0.7 -d 10 '
                    '-b {}-{}.val').format(organism1.org_id,
                                           organism2.org_id,
                                           organism1.org_id,
                                           organism2.org_id)

    utils.run_cmd(exec_command, cwd=cs.HUBALIGN_PATH)


def HubAlign_align(bio_net, file_path=cs.NETAL_PATH, check=True):
    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    file_name = ('{}.tab-{}.tab.alignment').format(organism1.org_id,
                                                   organism2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing HubAlign results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running HubAlign for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        HubAlign_run(bio_net)

        message = 'HubAlign finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.HUBALIGN_PATH, file_name), 'rU') as HubAlign:
        for line in HubAlign:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs


# MAGNA functions
def generate_gw_file_raw(organism, file_path=cs.CGRAAL_PATH):
    file_name = '{}.gw'.format(organism.org_id)

    if not os.path.exists(utils.join_path(file_path, file_name)):
        with open(utils.join_path(file_path, file_name), 'w') as gwfile:
            gwfile.write('# MAGNA GW\nLEDA.GRAPH\nstring\nshort\n')
            gwfile.write('{}\n'.format(organism.node_count))
            for i in range(organism.node_count):
                gwfile.write('|{' + organism.id_to_node[i] + '}|\n')
            gwfile.write('{}\n'.format(len(organism.edges)))
            for i in organism.edges:
                gwfile.write('{} {}'.format(i[0], i[1]) + ' 0 |{}|\n')


def generate_blast_sim_file(bio_net, file_path=cs.MAGNA_PATH):
    file_name = '{}-{}-sim.txt'.format(bio_net.org1.org_id,
                                       bio_net.org2.org_id)


    organism1 = bio_net.org1
    organism2 = bio_net.org2
    flip = False
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1
        flip = True

    bio_net.calculate_blast_matrix()

    if not os.path.exists(utils.join_path(file_path, file_name)):
        with open(utils.join_path(file_path, file_name), 'w') as simfile:
            simfile.write('{} {}\n'.format(organism1.node_count,
                                           organism2.node_count))
            for i in range(organism1.node_count):
                selection = []
                for j in range(organism2.node_count):
                    if flip:
                        selection.append(bio_net.blast_sim_n[bio_net.v_ind(j, i)])
                    else:
                        selection.append(bio_net.blast_sim_n[bio_net.v_ind(i, j)])
                simfile.write(' '.join(str(x) for x in selection) + '\n')


@utils.time_it
def MAGNA_run(bio_net, file_path=cs.MAGNA_PATH):
    # generate input info
    generate_gw_file_raw(bio_net.org1, file_path)
    generate_gw_file_raw(bio_net.org2, file_path)

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    # generate blast input
    generate_blast_sim_file(bio_net, file_path)

    exec_command = ('./magnapp_cli_linux64 -G {}.gw -H {}.gw -p {} -n {} -o '
                    '{}_{}-MAGNA -m EC -d {}-{}-sim.txt '
                    '-a {}').format(organism1.org_id,
                                    organism2.org_id,
                                    cs.MAGNA_P,
                                    cs.MAGNA_N,
                                    bio_net.org1.org_id,
                                    bio_net.org2.org_id,
                                    bio_net.org1.org_id,
                                    bio_net.org2.org_id,
                                    cs.MAGNA_ALPHA)

    utils.run_cmd(exec_command, cwd=cs.MAGNA_PATH)


def MAGNA_align(bio_net, file_path=cs.MAGNA_PATH, check=True):
    file_name = '{}_{}-MAGNA_final_alignment.txt'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing MAGNA results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running MAGNA for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        MAGNA_run(bio_net)

        message = 'MAGNA finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.MAGNA_PATH, file_name), 'rU') as MIGRAAL:
        for line in MIGRAAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])


                if pair is not None:
                    pairs.append(pair)

    return pairs


# PROPER functions
def generate_sim_files(organism, file_path=cs.PROPER_PATH):
    file_name = '{}.txt'.format(organism.org_id)
    with open(utils.join_path(file_path, file_name), 'w') as tabfile:
        for edge in organism.edges:
            es = organism.id_to_node[edge[0]]
            et = organism.id_to_node[edge[1]]
            tabfile.write(es + '\t' + et + '\n')


@utils.time_it
def PROPER_run(bio_net, file_path=cs.PROPER_PATH):
    # generate input info
    generate_sim_files(bio_net.org1, file_path)
    generate_sim_files(bio_net.org2, file_path)

    # generate eval files (blast)
    blast_xml_to_val(bio_net.org1, bio_net.org2, out_path=cs.PROPER_PATH)

    exec_command = ('./proper {}.txt {}.txt {}-{}.val {} {} '
                    '{}_{}-PROPER.txt').format(bio_net.org1.org_id,
                                               bio_net.org2.org_id,
                                               bio_net.org1.org_id,
                                               bio_net.org2.org_id,
                                               cs.PROPER_R,
                                               cs.PROPER_L,
                                               bio_net.org1.org_id,
                                               bio_net.org2.org_id)

    utils.run_cmd(exec_command, cwd=cs.PROPER_PATH)


def PROPER_align(bio_net, file_path=cs.PROPER_PATH, check=True):
    file_name = '{}_{}-PROPER.txt'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing PROPER results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running PROPER for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        PROPER_run(bio_net)

        message = 'PROPER finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.PROPER_PATH, file_name), 'rU') as MIGRAAL:
        for line in MIGRAAL:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs


# SPINAL functions
def generate_SPINAL_files(bio_net, file_path=cs.SPINAL_PATH):
    for organism in [bio_net.org1, bio_net.org2]:
        file_name = '{}.txt'.format(organism.org_id)
        if not os.path.exists(utils.join_path(file_path, file_name)):
            with open(utils.join_path(file_path, file_name), 'w') as tabfile:
                tabfile.write('INTERACTOR_A  INTERACTOR_B\n')
                for edge in organism.edges:
                    es = organism.id_to_node[edge[0]]
                    et = organism.id_to_node[edge[1]]
                    tabfile.write('{} {}\n'.format(es, et))

    # generate eval files (blast)
    blast_xml_to_val(bio_net.org1, bio_net.org2, out_path=cs.SPINAL_PATH)

    exec_command = ('python2 data_generation.py '
                    '{}.txt {}.txt {}-{}.val'.format(bio_net.org1.org_id,
                                                     bio_net.org2.org_id,
                                                     bio_net.org1.org_id,
                                                     bio_net.org2.org_id))

    utils.run_cmd(exec_command, cwd=cs.SPINAL_PATH)


@utils.time_it
def SPINAL_run(bio_net, algorithm, file_path=cs.SPINAL_PATH):
    # generate input
    generate_SPINAL_files(bio_net, file_path)

    out_file = '{}_{}-SPINAL-{}.out'.format(bio_net.org1.org_id,
                bio_net.org2.org_id, algorithm)

    org1_gml = '{}.txt.gml'.format(bio_net.org1.org_id)
    org2_gml = '{}.txt.gml'.format(bio_net.org2.org_id)


    if not os.path.exists(utils.join_path(file_path, out_file)):
        exec_command = ('./spinal.exe -{} -ns {} {} {}-{}.val.pin '
                        '{} {}').format(algorithm, org1_gml, org2_gml,
                        bio_net.org1.org_id, bio_net.org2.org_id,
                        out_file, cs.SPINAL_ALPHA)

        utils.run_cmd(exec_command, cwd=cs.SPINAL_PATH)


        exec_command = ('python2 spinal_original_names.py {} {} '
                        '{} {}_{}-SPINAL-{}.txt'.format(
                        org1_gml, org2_gml, out_file,
                        bio_net.org1.org_id, bio_net.org2.org_id,
                        algorithm))

        utils.run_cmd(exec_command, cwd=cs.SPINAL_PATH)


def SPINAL_align(bio_net, alg=cs.SPINAL_ALG, file_path=cs.SPINAL_PATH,
                 check=True):
    file_name = '{}_{}-SPINAL-{}.txt'.format(
        bio_net.org1.org_id, bio_net.org2.org_id, alg)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing SPINAL results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running SPINAL for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        SPINAL_run(bio_net, alg)

        message = 'SPINAL finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.SPINAL_PATH, file_name), 'rU') as SPINAL:
        for line in SPINAL:
            if line != '' and line[0] != '!':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                        elif (bio_net.similarity[bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1, id2, bio_net.similarity[bio_net.v_ind(id1, id2)])

                if pair is not None:
                    pairs.append(pair)

    return pairs


# Pathway analysis functions
org_codes = {
    '4932': 'SCE',
    '7227': 'DME',
    '9606': 'HSA',
}


def get_pathways(org, pw_file=cs.PW_FILE, save_path=cs.JSON_PATH):
    file_name = '{}.pathways.json'.format(org)
    file_path = utils.join_path(save_path, file_name)

    if (utils.file_exists(file_name, save_path)):
        message = 'pathways already stored for {}'.format(org)
        utils.print_log(message)

        return utils.load_json(file_path)

    else:
        message = 'parsing pathways from reactome for {}'.format(org)
        utils.print_log(message)

        # organism pathway dict
        pdict = {}

        # uniprot mapping
        mapping = string_db.get_uniprot_map(org)

        # org code
        code = org_codes[org]

        with open(pw_file, 'r') as index:
            while True:
                line = index.readline()
                line = line.strip()

                if line == '':
                    break

                try:
                    words = line.split('\t')
                    # UniProt identifier
                    uniprot_ac = words[0]
                    # Reactome Pathway Stable identifier
                    pathway_id = words[1]
                    pathway_org = pathway_id.split('-')[1]
                    pathway_code = pathway_id.split('-')[2]
                    # # Evidence Code
                    # pathway_code = words[4]
                    # string prot code
                    prot = mapping[uniprot_ac]
                    if pathway_org == code:
                        pdict[pathway_code] = pdict.get(pathway_code,
                                                        []) + [prot]
                except KeyError:
                    continue

        utils.write_json(pdict, file_path)

        return pdict


# optnet functions
def generate_optnet_files(bio_net, file_path=cs.OPTNET_PATH):
    for organism in [bio_net.org1, bio_net.org2]:
        file_name = '{}.net'.format(organism.org_id)

        with open(utils.join_path(file_path, file_name), 'w') as nfile:
            for edge in organism.edges:
                es = organism.id_to_node[edge[0]]
                et = organism.id_to_node[edge[1]]
                nfile.write('{} {}\n'.format(es, et))

    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    # generate eval files (blast) (o1-o2.val)
    blast_xml_to_val(organism1, organism2, out_path=cs.OPTNET_PATH)


@utils.time_it
def optnet_run(bio_net, file_path=cs.OPTNET_PATH):
    organism1 = bio_net.org1
    organism2 = bio_net.org2
    if organism1.node_count > organism2.node_count:
        organism1, organism2 = organism2, organism1

    # generate blast input
    generate_optnet_files(bio_net, file_path)

    exec_command = ('./optnetalign --net1 {}.net --net2 {}.net --total '
                    '--{} --bitscores {}-{}.val --blastsum --cxrate {} '
                    '--cxswappb {} --mutrate {} --mutswappb {} '
                    '--oneobjrate {} --dynparams --popsize {} '
                    '--generations {} --hillclimbiters {}  --timelimit {} '
                    '--outprefix {}-{} --finalstats >> '
                    '{}-{}.finalstats').format(
                        organism1.org_id,
                        organism2.org_id,
                        cs.optnet_alg,
                        organism1.org_id,
                        organism2.org_id,
                        cs.optnet_cxrate,
                        cs.optnet_cxswappb,
                        cs.optnet_mutrate,
                        cs.optnet_mutswappb,
                        cs.optnet_oneobjrate,
                        cs.optnet_popsize,
                        cs.optnet_generations,
                        cs.optnet_hillclimbiters,
                        cs.optnet_timelimit,
                        bio_net.org1.org_id,
                        bio_net.org2.org_id,
                        bio_net.org1.org_id,
                        bio_net.org2.org_id)

    utils.run_cmd(exec_command, cwd=cs.OPTNET_PATH)


def optnet_align(bio_net, file_path=cs.OPTNET_PATH, check=False):
    file_name = '{}-{}.finalstats'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing optnet results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running optnet for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        optnet_run(bio_net)

        message = 'optnet finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    prefix = '{}-{}'.format(
        bio_net.org1.org_id, bio_net.org2.org_id)
    runs = [int(x[x.rindex('_') + 1:-4]) for
            x in utils.list_files(cs.OPTNET_PATH)
            if x[-4:] == '.aln' and prefix in x]
    last_file = '{}_{}.aln'.format(prefix, max(runs))
    with open(utils.join_path(cs.OPTNET_PATH, last_file), 'rU') as optnet:
        for line in optnet:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs


# moduleAlign functions
def generate_moduleAlign_files(bio_net, file_path=cs.MODULEALIGN_PATH):
    for organism in [bio_net.org1, bio_net.org2]:
        file_name = '{}.net'.format(organism.org_id)

        with open(utils.join_path(file_path, file_name), 'w') as nfile:
            for edge in organism.edges:
                es = organism.id_to_node[edge[0]]
                et = organism.id_to_node[edge[1]]
                nfile.write('{} {}\n'.format(es, et))

    # generate eval files (blast) (o1-o2.val)
    blast_xml_to_val(bio_net.org1, bio_net.org2, out_path=cs.MODULEALIGN_PATH)
    val_name = '{}-{}.val'.format(bio_net.org1.org_id, bio_net.org2.org_id)
    sim_name = '{}-{}.blast'.format(bio_net.org1.org_id, bio_net.org2.org_id)
    utils.rename_file(val_name, sim_name, path=cs.MODULEALIGN_PATH)


@utils.time_it
def moduleAlign_run(bio_net, file_path=cs.MODULEALIGN_PATH):
    # generate blast input
    generate_moduleAlign_files(bio_net, file_path)

    exec_command = ('./script.sh {} {} {}').format(
        bio_net.org1.org_id,
        bio_net.org2.org_id,
        cs.moduleAlign_alpha)

    utils.run_cmd(exec_command, cwd=cs.MODULEALIGN_PATH)


def moduleAlign_align(bio_net, file_path=cs.MODULEALIGN_PATH, check=False):
    file_name = '{}-{}-a{}.alignment'.format(
        bio_net.org1.org_id, bio_net.org2.org_id,
        cs.moduleAlign_alpha)

    if (check and utils.file_exists(file_name, file_path)):
        message = 'using existing moduleAlign results for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    else:
        message = 'running moduleAlign for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

        moduleAlign_run(bio_net)

        message = 'moduleAlign finished for {}-{}'.format(
            bio_net.org1.org_id, bio_net.org2.org_id)
        utils.print_log(message)

    pairs = []
    with open(utils.join_path(cs.MODULEALIGN_PATH, file_name), 'rU') as optnet:
        for line in optnet:
            if line != '':
                prots = line.split()
                pr1 = [bio_net.org1.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org1.org_id))]
                pr2 = [bio_net.org2.node_to_id[x]
                       for x in prots if x.startswith(
                    str(bio_net.org2.org_id))]
                pair = None
                for id1 in pr1:
                    for id2 in pr2:
                        if pair is None:
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                        elif (bio_net.similarity[
                                bio_net.v_ind(id1, id2)] > pair[2]):
                            pair = (id1,
                                    id2,
                                    bio_net.similarity[bio_net.v_ind(id1,
                                                                     id2)])
                if pair is not None:
                    pairs.append(pair)

    return pairs
