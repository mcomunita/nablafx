import argparse
from frechet_audio_distance import FrechetAudioDistance


parser = argparse.ArgumentParser()

# add PROGRAM level args
parser.add_argument("--model", type=str, default="vggish", help="vggish, pann")
parser.add_argument("--use_pca", action="store_true")
parser.add_argument("--use_activation", action="store_true")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--bkg_dir", type=str)
parser.add_argument("--eval_dir", type=str)
parser.add_argument("--bkg_emb_dir", type=str, help="save bkg emb to this dir")
parser.add_argument("--eval_emb_dir", type=str, help="save eval emb to this dir")

args = parser.parse_args()

if __name__ == "__main__":

    print(args.model)
    print(args.use_pca)
    print(args.use_activation)
    print(args.verbose)
    print(args.bkg_dir)
    print(args.eval_dir)
    print(args.bkg_emb_dir)
    print(args.eval_emb_dir)

    # define frechet audio distance model
    frechet = FrechetAudioDistance(
        args.model,
        args.use_pca,
        args.use_activation,
        args.verbose,
    )
    
    fad_score = frechet.score(
        args.bkg_dir,
        args.eval_dir,
    )

    print("FAD score: ", fad_score)