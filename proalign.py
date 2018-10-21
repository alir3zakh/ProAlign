"""
this module contains codes that are responsible to do the main logical flow
and to manage the overall data flow from reading inputs, processing them,
generating variety of alignment scores if need be and take care of
visualizations and conclusions
"""

import argparse
import gc

import utils
import string_db
import interface
import organism
import align
import visualize
import constants as cs

# list of all acceptible options
algorithms = ['greedy', 'sgreedy', 'clstr', 'max', 'rclst',
              'cclst', 'l2clstr', 'isoN', 'NETAL', 'pinalog',
              'l2extend', 'l2selextend', 'l2mincpl', 'l2mincplextend',
              'l2mincplselextend', 'l2maxcut', 'l2maxcutextend',
              'l2maxcutselextend', 'l2brutecut', 'l2brutecutextend',
              'l2brutecutselextend', 'seedex', 'seedexneigh', 'seedexcost',
              'CGRAAL', 'GRAAL', 'MIGRAAL', 'HubAlign', 'MAGNA', 'PROPER',
              'SPINAL-I', 'SPINAL-II', 'seedexblast', 'seedexsingle',
              'seedexsingle-jac', 'seedexsingle-ada', 'seedex-greedy',
              'seedexproper', 'seedexsingle-jacn', 'seedexsingle-adan',
              'seedexsingle-jacnn', 'seedexsingle-adann', 'optnet',
              'moduleAlign']


algorithms = ['CGRAAL', 'MIGRAAL', 'NETAL', 'HubAlign', 'PROPER',
              'GRAAL', 'seedexsingle', 'seedexsingle-jac', 'seedexsingle-ada',
              'seedexsingle-prd', 'seedexsingle-prd-jac', 'seedexsingle-prd-ada',
              'seedexsingle-prb', 'seedexsingle-prb-jac', 'seedexsingle-prb-ada',
              'seedexSLE', 'seedexSLE-jac', 'seedexSLE-ada',
              'seedexBC', 'seedexBC-jac', 'seedexBC-ada',
              'seedexCFBC', 'seedexCFBC-jac', 'seedexCFBC-ada',
              'seedexCFCC', 'seedexCFCC-jac', 'seedexCFCC-ada',
              'seedexCC', 'seedexCC-jac', 'seedexCC-ada',
              'SPINAL-I', 'SPINAL-II', 'pinalog', 'MAGNA', 'isoN',
              'seedex-SA', 'optnet', 'moduleAlign']


similarity_modes = ['raw_blast', 'blast_power', 'just_power', 'no_sim',
                    'rel_blast']


# function to initialize network
@utils.time_it
def initialize_network(organism_ids, align_method, similarity_mode,
                       power_alpha=cs.ALPHA_BIAS):

    # sort ids to fix order
    organism_ids.sort()

    # check if initial files exist
    list(map(string_db.check_initial_files, organism_ids))

    # create blast db for organisms
    list(map(interface.create_blast_db, organism_ids))

    # print('\n\n\n!!!\n\n\n')
    print(organism_ids)

    # run blastp scores if needed
    interface.run_blast_prot(*organism_ids)
    # if (align_method in ['isoN', 'NETAL']) or (similarity_mode == 'rel_blast'):
    for x in organism_ids:
        interface.run_blast_prot(x, x)

    # parse organism ppi networks from input
    org1, org2 = map(string_db.parse_organism, organism_ids)

    # create bio_net object with propper options
    bio_net = organism.BioNet(org1, org2, similarity_mode, power_alpha)

    return bio_net


# function to process whole alignment flow
@utils.time_it
def process(organism_ids, align_method, similarity_mode,
            power_alpha=cs.ALPHA_BIAS, check=True, visual=False):

    # load bio_net object
    bio_net = initialize_network(organism_ids, align_method,
                                 similarity_mode, power_alpha)

    # create aligner object
    aligner = align.Aligner(align_method)

    alignment = aligner.align(bio_net, check=check)

    # visualization
    if visual:
        visualize.gephi_organism_ppi(bio_net.org1)
        visualize.gephi_organism_ppi(bio_net.org2)
        visualize.gephi_network_aligned(alignment, bio_net)
        visualize.gephi_network_aligned_comp(alignment, bio_net)

    return alignment


@utils.time_it
def analyze_similarity(organism_ids, align_method,
                       similarity_mode, power_alpha=cs.ALPHA_BIAS):

    # load bio_net object
    bio_net = initialize_network(organism_ids, align_method,
                                 similarity_mode, power_alpha)

    # visualize blast info
    visualize.sim_degree(bio_net)
    visualize.sim_degree_3d(bio_net)


@utils.time_it
def visualize_organism(organism_id):
    # check if initial files exist
    string_db.check_initial_files(organism_id)
    org = string_db.parse_organism(organism_id)

    # visualize with gephi
    visualize.gephi_organism_ppi(org)


def main():
    if args.algo == 'all':
        algs = [x for x in algorithms if x != 'max']
    elif args.algo in algorithms:
        algs = [args.algo]
    else:
        raise Exception(('algorithm name not valid, '
                         'valid options are: {}').format(algorithms))

    if args.similarity in similarity_modes:
        modes = [args.similarity]
    else:
        raise Exception(('algorithm name not valid, '
                         'valid options are: {}').format(similarity_modes))

    organism_ids = [args.organism_id1, args.organism_id2]

    for alg in algs:
        for mode in modes:
            gc.collect()
            process(organism_ids, alg, mode, args.power_alpha,
                    args.check, args.visual)


if __name__ == "__main__":
    # use argparse to verify input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("organism_id1", type=str, help="First organism's ID")
    parser.add_argument("organism_id2", type=str, help="Second organism's ID")
    parser.add_argument("--algo", type=str, default='all',
                        help="choose the alignment method")
    parser.add_argument("--similarity", type=str, default='raw_blast',
                        help="choose the similarity measure used for alignment")
    parser.add_argument("--power_alpha", type=float, default=cs.ALPHA_BIAS,
                        help="choose the power method alpha (damping factor)")
    parser.add_argument("--check", dest='check', action='store_true',
                        help="check for existing calculations")
    parser.add_argument("--no-check", dest='check', action='store_false',
                        help="don't check for existing calculations")
    parser.set_defaults(check=True)
    parser.add_argument("--visualize", dest='visual', action='store_true',
                        help="visualize alignment")
    parser.set_defaults(visual=False)
    args = parser.parse_args()

    main()
