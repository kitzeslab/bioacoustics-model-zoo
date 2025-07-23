from github import Github


def list_hawkears_checkpoints(tag, lowband=False, embedding=False):

    if lowband:  # classifier for low frequency sounds like ruffed grouse
        search_path = "data/"
        pattern = "low_band.ckpt"
    elif embedding:  # embedding model
        search_path = "data/ckpt-search/"
        pattern = ".ckpt"
    else:  # typical bird species classifier ensemble
        search_path = "data/ckpt/"
        pattern = ".ckpt"

    # Replace with your GitHub token if needed (for higher rate limits)
    g = Github()

    repo = g.get_repo("jhuus/HawkEars")
    ref = f"tags/{tag}"

    # Get the SHA for the tag
    tag_ref = repo.get_git_ref(ref)
    commit_sha = tag_ref.object.sha

    # Get the tree at the commit
    tree = repo.get_git_tree(commit_sha, recursive=True)

    # Filter for .ckpt files in data/ckpt/
    ckpt_files = [
        f"https://github.com/jhuus/HawkEars/raw/refs/tags/{tag}/" + f.path
        for f in tree.tree
        if f.path.startswith(search_path) and f.path.endswith(pattern)
    ]

    return ckpt_files
