
def _add(lines: list[str], *, title=None, line=None):
    if title:
        lines.append("\033[1m" + title + "\033[0m")
    if line:
        lines.append(line)

def format_args(args) -> str:
    lines = []

    _add(lines, title="Basic Config")
    _add(lines, line=f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    _add(lines, line=f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    lines.append("")

    _add(lines, title="Data Loader")
    _add(lines, line=f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    _add(lines, line=f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    _add(lines, line=f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    _add(lines, line=f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    lines.append("")

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        _add(lines, title="Forecasting Task")
        _add(lines, line=f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        _add(lines, line=f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        _add(lines, line=f'  {"Inverse:":<20}{args.inverse:<20}')
        lines.append("")

    if args.task_name == 'imputation':
        _add(lines, title="Imputation Task")
        _add(lines, line=f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        lines.append("")

    if args.task_name == 'anomaly_detection':
        _add(lines, title="Anomaly Detection Task")
        _add(lines, line=f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        lines.append("")

    _add(lines, title="Model Parameters")
    _add(lines, line=f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    _add(lines, line=f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    _add(lines, line=f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    _add(lines, line=f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    _add(lines, line=f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    _add(lines, line=f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    _add(lines, line=f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    _add(lines, line=f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    lines.append("")

    _add(lines, title="Run Parameters")
    _add(lines, line=f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    _add(lines, line=f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    _add(lines, line=f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    _add(lines, line=f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    _add(lines, line=f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    lines.append("")

    _add(lines, title="GPU")
    _add(lines, line=f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    _add(lines, line=f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    lines.append("")

    _add(lines, title="De-stationary Projector Params")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    _add(lines, line=f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    lines.append("")

    return "\n".join(lines)


def print_args(args):
    print(format_args(args))
