from Bio.Align import PairwiseAligner
from Bio.SeqIO import parse
from tqdm import tqdm

dude_fasta=list(parse("/drug/rag/DUD_E.fasta","fasta"))
pdbbind_fasta=list(parse("/drug/rag/PDBBind.fasta","fasta"))

aligner = PairwiseAligner()

# 计算 pairwise identity
results = []
for dude_seq in tqdm(dude_fasta):
    for pdbbind_seq in tqdm(pdbbind_fasta):
        alignment = aligner.align(dude_seq.seq, pdbbind_seq.seq)[0]

        seq_len=min(len(dude_seq.seq),len(pdbbind_seq.seq))
        score=alignment.score/seq_len
        results.append((dude_seq.id,pdbbind_seq.id,score))

# 输出或保存结果
print(results)



output_file =  "/drug/rag/pairwise_alignment_results.txt"
with open (output_file,"w") as f:
    for item in results:
        f.write(item[0]+"\t"+item[1]+"\t"+str(round(item[2],3))+"\n")