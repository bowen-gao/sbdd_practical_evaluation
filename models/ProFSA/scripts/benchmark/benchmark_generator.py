import os
from dataset import BioLipDataset, PDBBindDataset, DUDEDataset
from Filter import SequenceSimilarityFilter, FLAPPFilter, MorganFilter

class BenchmarkGenerator:

    def __init__(self):
        pass

    def run(self):
        BioLip_dataset=BioLipDataset()
        dude_dataset=DUDEDataset()
        pdbbind_dataset=PDBBindDataset()
        
        # FLAPP_filter=FLAPPFilter(dataset_s=dude_dataset,
        #                         dataset_t=pdbbind_dataset,
        #                         FLAPP_output_file="/data/rag/FLAPP/dude_pdbbind_output.txt")
        # FLAPP_filter.run(thresholds=[0.6,0.9])

        # Morgan_filter=MorganFilter(dataset_s=dude_dataset,
        #                             dataset_t=BioLip_dataset,
        #                             morgan_output_file="/data/rag/morgan_dude_BioLip_result.txt")
        # Morgan_filter.run(sizes=[20000,30000,40000,50000])


        FLAPP_filter=FLAPPFilter(dataset_s=dude_dataset,
                                dataset_t=BioLip_dataset,
                                FLAPP_output_file="/data/rag/FLAPP/dude_BioLip_output.txt")
        FLAPP_filter.run(thresholds=[0.6,0.9])

        # seq_filter=SequenceSimilarityFilter(dataset_s=dude_dataset,
        #                                     dataset_t=BioLip_dataset,
        #                                     similarity_output_file="/data/rag/SeqSimilarity_result.txt")
        # seq_filter.run(sizes=[20000,30000,40000,50000])


benchmark_generator=BenchmarkGenerator()
benchmark_generator.run()
