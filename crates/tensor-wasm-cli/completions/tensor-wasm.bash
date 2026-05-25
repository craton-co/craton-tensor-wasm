_tensor-wasm() {
    local i cur prev opts cmd
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmd=""
    opts=""

    for i in "${COMP_WORDS[@]:0:COMP_CWORD}"; do
        case "${cmd},${i}" in
            ",$1")
                cmd="tensor__wasm"
                ;;
            tensor__wasm,bench)
                cmd="tensor__wasm__bench"
                ;;
            tensor__wasm,completions)
                cmd="tensor__wasm__completions"
                ;;
            tensor__wasm,deploy)
                cmd="tensor__wasm__deploy"
                ;;
            tensor__wasm,invoke)
                cmd="tensor__wasm__invoke"
                ;;
            tensor__wasm,man)
                cmd="tensor__wasm__man"
                ;;
            tensor__wasm,metrics)
                cmd="tensor__wasm__metrics"
                ;;
            tensor__wasm,observe)
                cmd="tensor__wasm__observe"
                ;;
            tensor__wasm,run)
                cmd="tensor__wasm__run"
                ;;
            tensor__wasm,snapshot)
                cmd="tensor__wasm__snapshot"
                ;;
            tensor__wasm__snapshot,save)
                cmd="tensor__wasm__snapshot__save"
                ;;
            tensor__wasm__snapshot,restore)
                cmd="tensor__wasm__snapshot__restore"
                ;;
            *)
                ;;
        esac
    done

    case "${cmd}" in
        tensor__wasm)
            opts="-h --help -V --version --tenant run deploy invoke bench snapshot metrics observe completions man help"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 1 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --tenant)
                    COMPREPLY=()
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__run)
            opts="-h --help --export --args --tenant <FILE>"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --export --args --tenant" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --export|--args|--tenant)
                    COMPREPLY=()
                    return 0
                    ;;
                *)
                    COMPREPLY=( $(compgen -f -- "${cur}") )
                    ;;
            esac
            return 0
            ;;
        tensor__wasm__deploy)
            opts="-h --help --server --name --tenant <FILE>"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --server --name --tenant" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --server|--name|--tenant)
                    COMPREPLY=()
                    return 0
                    ;;
                *)
                    COMPREPLY=( $(compgen -f -- "${cur}") )
                    ;;
            esac
            return 0
            ;;
        tensor__wasm__invoke)
            opts="-h --help --server --args --tenant <ID>"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --server --args --tenant" -- "${cur}") )
                return 0
            fi
            return 0
            ;;
        tensor__wasm__bench)
            opts="-h --help --export --n --tenant <FILE>"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --export --n --tenant" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --export|--n|--tenant)
                    COMPREPLY=()
                    return 0
                    ;;
                *)
                    COMPREPLY=( $(compgen -f -- "${cur}") )
                    ;;
            esac
            return 0
            ;;
        tensor__wasm__snapshot)
            opts="-h --help --tenant save restore help"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__snapshot__save)
            opts="-h --help --instance --output --server --tenant"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__snapshot__restore)
            opts="-h --help --input --as-instance --server --max-decompressed --tenant"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__metrics)
            opts="-h --help --server --tenant"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__observe)
            opts="-h --help --addr --interval --tenant"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        tensor__wasm__completions)
            opts="-h --help --out-dir --tenant bash zsh fish powershell elvish"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --out-dir --tenant" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --out-dir)
                    COMPREPLY=( $(compgen -d -- "${cur}") )
                    return 0
                    ;;
                --tenant)
                    COMPREPLY=()
                    return 0
                    ;;
                *)
                    COMPREPLY=( $(compgen -W "bash zsh fish powershell elvish" -- "${cur}") )
                    ;;
            esac
            return 0
            ;;
        tensor__wasm__man)
            opts="-h --help --out-dir --tenant"
            if [[ ${cur} == -* ]] ; then
                COMPREPLY=( $(compgen -W "-h --help --out-dir --tenant" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --out-dir)
                    COMPREPLY=( $(compgen -d -- "${cur}") )
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            return 0
            ;;
    esac
}

if [[ "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -ge 4 || "${BASH_VERSINFO[0]}" -gt 4 ]]; then
    complete -F _tensor-wasm -o nosort -o bashdefault -o default tensor-wasm
else
    complete -F _tensor-wasm -o bashdefault -o default tensor-wasm
fi
