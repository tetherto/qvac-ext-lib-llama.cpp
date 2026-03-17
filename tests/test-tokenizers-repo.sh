#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x $toktest ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

if [ -d $folder ] && [ -d $folder/.git ]; then
    (cd $folder; git pull)
else
    GIT_LFS_SKIP_SMUDGE=1 git clone $repo $folder
fi

shopt -s globstar
for gguf in $folder/**/*.gguf; do
    if head -c 4 "$gguf" | grep -q 'GGUF'; then continue; fi
    rel="${gguf#$folder/}"
    printf "Downloading LFS file via curl: %s\n" "$rel"
    curl -fL -o "$gguf" "$repo/resolve/main/$rel"
done

if [ "$(uname -m)" = s390x ]; then
    for f in $folder/*/*.gguf; do
        echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
    done
fi

fail=0
pass=0

for gguf in $folder/**/*.gguf; do
    if [ -f $gguf.inp ] && [ -f $gguf.out ]; then
        if ! head -c 3 "$gguf" | grep -q 'GGUF'; then
            printf "WARNING: $gguf is not a valid GGUF file (LFS pointer or corrupt), skipping...\n"
            continue
        fi
        if $toktest $gguf; then
            pass=$((pass + 1))
        else
            printf "FAILED: $gguf\n"
            fail=$((fail + 1))
        fi
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done

printf "\nResults: $pass passed, $fail failed\n"
if [ $fail -gt 0 ]; then
    exit 1
fi

