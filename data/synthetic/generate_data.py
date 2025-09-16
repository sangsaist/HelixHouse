import argparse  #Command-line argument parsing
import random    #Random number generation
from pathlib import Path  #File path handling
import pandas as pd  # Data manipulation and CSV output


TOTAL = 2000 # Total no of synthetic sequences to generate
UNASSIGNED_FRAC = 0.28 # Fraction of sequences that will be unassigned
REF_FRAC = 0.10 # Fraction of sequences that will be reference sequences



# List of sample site names
SAMPLES = [
    "Canyon_A","Canyon_B","Trench_A","Trench_B",
    "Vent_A","Vent_B","Seamount_A","Seamount_B",
    "AbyssalPlain_A","AbyssalPlain_B","Slope_A","Slope_B"
]


# Depth ranges for each sample site
SITE_DEPTH = {
    "Canyon_A": (1000,1500), "Canyon_B": (1200,1600),
    "Trench_A": (4000,6000), "Trench_B": (4500,6000),
    "Vent_A": (1800,2600), "Vent_B": (2000,3200),
    "Seamount_A": (900,1600), "Seamount_B": (800,2000),
    "AbyssalPlain_A": (3000,3800), "AbyssalPlain_B": (3200,4200),
    "Slope_A": (600,1000), "Slope_B": (700,1100)
}


# List of environmental data sources
ENV_SRC = ["DeepSEARCH","Mendeley","CCLME","USGS"]


# Known taxa for each marker
KNOWN_18S = [
    "DeepSea_Protist_A","DeepSea_Protist_B","ColdWater_Coral_A","ColdWater_Coral_B",
    "Bathyal_Annelid_A","Hydrozoan_Cnidaria_A","Planktonic_Eukaryote_A","Planktonic_Eukaryote_B",
    "DeepFungus_A","Echinoderm_Benthic_B"
]

KNOWN_COI = [
    "Bathypelagic_Fish_A","Bathypelagic_Fish_B","Bathypelagic_Fish_C",
    "Hydrothermal_Shrimp_A","Hydrothermal_Shrimp_B",
    "Abyssal_Crab_A","Abyssal_Crab_B","Deep_Mollusk_A","Deep_Cephalopod_A","Echinoderm_Benthic_A",
    "Bathypelagic_Fish_D","Hydrothermal_Shrimp_C"
]


# Sequence length ranges for synthetic and reference sequences
L_18S = (220,260)
L_COI = (620,680)
REF_18S = (240,260)
REF_COI = (640,660)


# Helper Functions
def rseq(n, rng):
    """Generate a random DNA sequence of length n."""
    return ''.join(rng.choice("ATCG") for _ in range(n))

def rc_for(taxon, rng):
    """Return a simulated read count for a given taxon."""
    if any(x in taxon for x in ("Bathypelagic_Fish_A","DeepSea_Protist_A","Bathypelagic_Fish_B")):
        return rng.randint(80,350)
    if any(x in taxon for x in ("ColdWater_Coral","Abyssal_Crab","Hydrothermal_Shrimp_A")):
        return rng.randint(30,150)
    if any(x in taxon for x in ("Plankton","DeepSea_Protist_B","Deep_Mollusk")):
        return rng.randint(10,80)
    return rng.randint(1,40)

def conf_k(marker, rng):
    """Return a confidence score for a known taxon."""
    return round(rng.uniform(0.70,0.95),2) if marker=="18S" else round(rng.uniform(0.72,0.96),2)

def conf_ref(rng):
    """Return a confidence score for a reference sequence."""
    return round(rng.uniform(0.95,0.999),3)

def mk(sid, seq, marker, tax, src, sample, rc, lvl, conf, loc, depth):
    """Create a dictionary representing a single sequence record."""
    return {
        "seq_id": sid,
        "Sequence": seq,
        "Marker": marker,
        "Taxonomy": tax,
        "Source": src,
        "Sample_ID": sample,
        "Read_Count": rc,
        "Taxonomic_Level": lvl,
        "Confidence_Score": conf,
        "Location": loc,
        "Depth_m": depth
    }


# Generator code
# Generator function for synthetic eDNA data
def gen(total=TOTAL, seed=42):
    rng = random.Random(seed)  # Random number generator
    total = int(total)
    n_un = int(total * UNASSIGNED_FRAC)  # Number of unassigned sequences
    n_known = total - n_un  # Number of known sequences
    n18 = total // 2  # Number of 18S marker sequences
    ncoi = total - n18  # Number of COI marker sequences

    recs = []  # List to store all sequence records
    cnt = 1
    # Generate 18S marker sequences
    for _ in range(n18):
        sid = f"18S_{cnt:05d}"; cnt += 1
        seq = rseq(rng.randint(*L_18S), rng)
        src = rng.choice(ENV_SRC)
        sample = rng.choice(SAMPLES)
        depth = rng.randint(*SITE_DEPTH[sample])
        recs.append(mk(sid, seq, "18S", None, src, sample, None, None, None, sample, depth))

    # Generate COI marker sequences
    for _ in range(ncoi):
        sid = f"COI_{cnt:05d}"; cnt += 1
        seq = rseq(rng.randint(*L_COI), rng)
        src = rng.choice(ENV_SRC)
        sample = rng.choice(SAMPLES)
        depth = rng.randint(*SITE_DEPTH[sample])
        recs.append(mk(sid, seq, "COI", None, src, sample, None, None, None, sample, depth))

    rng.shuffle(recs)  # Shuffle records for randomness

    idxs = list(range(total))
    rng.shuffle(idxs)
    known_idxs = set(idxs[:n_known])  # Indices for known sequences
    un_idxs = set(idxs[n_known:])     # Indices for unassigned sequences

    # Assign known taxa and confidence scores
    for i in known_idxs:
        rec = recs[i]
        if rec["Marker"] == "18S":
            tax = rng.choice(KNOWN_18S)
            conf = conf_k("18S", rng)
        else:
            tax = rng.choice(KNOWN_COI)
            conf = conf_k("COI", rng)
        rc = rc_for(tax, rng)
        rec.update({"Taxonomy": tax, "Read_Count": rc, "Taxonomic_Level": "species", "Confidence_Score": conf})

    # Mark unassigned sequences
    for i in un_idxs:
        rec = recs[i]
        rec.update({"Taxonomy": "Unassigned", "Taxonomic_Level": "unknown", "Confidence_Score": None})
        rec["Read_Count"] = rng.randint(1,30) if rec["Source"] in ENV_SRC else None

    # Replace some records with reference sequences
    n_ref = int(total * REF_FRAC)
    ref_pos = rng.sample(range(total), n_ref)
    for pos in ref_pos:
        marker = rng.choice(["18S","COI"])
        sid = f"REF_{marker}_{rng.randint(10000,99999)}"
        if marker == "18S":
            seq = rseq(rng.randint(*REF_18S), rng)
            tax = rng.choice(KNOWN_18S)
            src = "SILVA"
        else:
            seq = rseq(rng.randint(*REF_COI), rng)
            tax = rng.choice(KNOWN_COI)
            src = "BOLD"
        recs[pos] = mk(sid, seq, marker, tax, src, None, None, "reference", conf_ref(rng), None, None)

    return recs

# Command-line interface for running the script
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=TOTAL)  # Number of sequences to generate
    p.add_argument("--seed", type=int, default=42)      # Random seed for reproducibility
    p.add_argument("--out", type=Path, default=Path.cwd() / "synthetic_edna.csv")  # Output CSV file path
    args = p.parse_args()

    print("Generating synthetic eDNA...")
    rows = gen(total=args.total, seed=args.seed)  # Generate synthetic data
    df = pd.DataFrame(rows)  # Convert to DataFrame
    cols = ["seq_id","Sequence","Marker","Taxonomy","Source","Sample_ID",
            "Read_Count","Taxonomic_Level","Confidence_Score","Location","Depth_m"]
    args.out.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    df.to_csv(args.out, index=False, columns=cols)  # Save to CSV
    print(f"Saved {len(df)} rows to {args.out}")
    print("Markers:", df["Marker"].value_counts().to_dict())  # Print marker counts
    print("Unassigned:", int((df["Taxonomy"]=="Unassigned").sum()))  # Print unassigned count
    print("\nSample:")
    # Print a sample of the generated data
    print(df.sample(min(8,len(df)), random_state=args.seed)[["seq_id","Marker","Taxonomy","Source","Sample_ID","Read_Count","Confidence_Score"]])

# Run main if script is executed directly
if _name_ == "_main_":
    main()