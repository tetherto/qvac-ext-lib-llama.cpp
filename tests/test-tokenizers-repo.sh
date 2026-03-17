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

git lfs install 2>/dev/null

if [ -d $folder ] && [ -d $folder/.git ]; then
    (cd $folder; git pull && git lfs pull)
else
    git clone $repo $folder
    (cd $folder; git lfs pull)

    # byteswap models if on big endian
    if [ "$(uname -m)" = s390x ]; then
        for f in $folder/*/*.gguf; do
            echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
        done
    fi
fi

fail=0
pass=0

shopt -s globstar
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

