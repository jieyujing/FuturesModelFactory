#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import clickhouse_connect  # type: ignore[import-untyped]
import polars as pl
import yaml

from futures_model_factory.utils.schema import EXCHANGE_SUFFIX_MAP, normalize_code_expr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 ZIP 压缩包或整个目录下的期货 1m 分钟线数据导入 ClickHouse。"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="/Users/link/Downloads/AP_all_contracts_20200101_to_20260605_1m.zip",
        help="ZIP 压缩包的物理路径，或者是存放多个 ZIP 的目录路径",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/futures_smoke.yaml",
        help="包含数据库配置的 YAML 配置文件路径",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="限制每个 ZIP 包中导入 of CSV 文件数量（用于测试/冒烟）",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="仅刷新数据索引表（合约元数据范围和行数），不进行数据导入",
    )
    return parser.parse_args()


def load_db_config(config_path: str | Path) -> dict[str, any]:
    """从 YAML 配置文件中读取 ClickHouse 数据库 DSN 参数。"""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_config = config.get("database")
    if not db_config:
        raise ValueError(f"配置文件中未找到 'database' 配置节点: {p}")

    required_keys = {"host", "port", "username", "password", "database"}
    missing = required_keys - set(db_config)
    if missing:
        raise ValueError(f"数据库配置缺失必要项: {sorted(missing)}")

    return db_config


def init_clickhouse_table(client: any) -> None:
    """在 ClickHouse 中初始化目标大表。"""
    # 采用 toYear(datetime) 分区，(code, datetime) 联合主键排序的 DDL
    ddl = """
    CREATE TABLE IF NOT EXISTS default.futures_1m_bar (
        datetime DateTime64(0),
        code String,
        open Float64,
        high Float64,
        low Float64,
        close Float64,
        volume Float64,
        open_interest Float64
    ) ENGINE = MergeTree()
    PARTITION BY toYear(datetime)
    ORDER BY (code, datetime)
    """
    client.command(ddl)
    print("ClickHouse 目标表 'default.futures_1m_bar' 初始化/检查完毕。")


def refresh_data_index(client: any) -> None:
    """通过主表聚合，刷新数据索引表（合约范围、行数等元数据）。"""
    print("\n正在生成/刷新数据索引表 'default.futures_1m_bar_index'...")
    ddl = """
    CREATE TABLE IF NOT EXISTS default.futures_1m_bar_index (
        code String,
        min_datetime DateTime64(0),
        max_datetime DateTime64(0),
        total_rows UInt64,
        updated_at DateTime DEFAULT now()
    ) ENGINE = ReplacingMergeTree(updated_at)
    ORDER BY code
    """
    client.command(ddl)

    # 1. 临时截断索引表，防止有旧记录无法覆盖
    client.command("TRUNCATE TABLE default.futures_1m_bar_index")

    # 2. 从主表 GROUP BY 并重新插入
    refresh_sql = """
    INSERT INTO default.futures_1m_bar_index (code, min_datetime, max_datetime, total_rows)
    SELECT
        code,
        min(datetime) as min_datetime,
        max(datetime) as max_datetime,
        count() as total_rows
    FROM default.futures_1m_bar
    GROUP BY code
    """
    client.command(refresh_sql)
    print("数据索引表 'default.futures_1m_bar_index' 刷新成功！")


def clean_and_normalize_data(df: pl.DataFrame, code: str) -> pl.DataFrame:
    """清洗与规范化 Polars DataFrame，以匹配目标 ClickHouse 表结构。"""
    # 1. 确保列名大写转换或按指定字段提取
    required_cols = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 数据中缺失必要列: {sorted(missing)}")

    # 2. 字段类型转换、追加 code 并使用 schema 模块自动清洗官方代码后缀
    cleaned = df.with_columns(
        pl.col("datetime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
        pl.lit(code).cast(pl.String).alias("code"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("open_interest").cast(pl.Float64),
    ).with_columns(normalize_code_expr("code").alias("code"))

    # 3. 按目标表的列顺序精确 select
    target_cols = [
        "datetime",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]
    return cleaned.select(target_cols)


def safe_insert_df(client: any, table: str, df: pl.DataFrame, max_retries: int = 5) -> None:
    """带指数退避的 ClickHouse 写入，自动应对网络抖动、Cloudflare 502/524 限制或数据库瞬时过载。"""
    total_len = len(df)
    chunk_size = 100000

    chunks = []
    if total_len > chunk_size:
        for offset in range(0, total_len, chunk_size):
            chunks.append(df.slice(offset, chunk_size))
    else:
        chunks.append(df)

    for chunk_idx, df_chunk in enumerate(chunks, start=1):
        pdf = df_chunk.to_pandas()
        retry_delay = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                client.insert_df(table, pdf)
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"写入分片 [{chunk_idx}/{len(chunks)}] 在 {max_retries} 次尝试后依然失败。")
                    raise e
                print(
                    f"写入分片 [{chunk_idx}/{len(chunks)}] 失败 (尝试 {attempt}/{max_retries})，"
                    f"原因: {e}。将在 {retry_delay:.1f} 秒后重试..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2.0


def import_single_zip(
    client: any, zip_path: Path, limit_files: int | None = None
) -> int:
    """导入单个 ZIP 压缩包中的所有分钟线数据，返回导入的总行数。"""
    print("\n==========================================")
    print(f"开始处理 ZIP 压缩包: {zip_path.name}")
    print("==========================================")
    start_time = time.time()
    total_rows = 0

    symbol = zip_path.name.split("_")[0].upper()
    with zipfile.ZipFile(zip_path, "r") as zf:
        import re
        if symbol == "TF":
            # TF 压缩包特殊处理：顺带导入孤儿品种 IF
            pattern = re.compile(r"^(TF|IF)(?:\d+|8888|9999)")
        else:
            pattern = re.compile(rf"^{symbol}(?:\d+|8888|9999)")
        all_csvs = [name for name in zf.namelist() if name.endswith(".csv")]
        csv_files = [
            name for name in all_csvs
            if pattern.match(name.split("-")[0].upper())
        ]
        match_desc = f"{symbol}+IF" if symbol == "TF" else symbol
        print(f"ZIP 压缩包中总计有 {len(all_csvs)} 个 CSV 文件，其中属于品种 {match_desc} 的合约有 {len(csv_files)} 个。")

        if limit_files is not None:
            csv_files = csv_files[:limit_files]
            print(f"已应用 --limit-files 限制，仅处理前 {len(csv_files)} 个合约。")

        # 3.1 预先批量清理已有 code 数据，规避高频并发 DELETE Mutations 导致 ClickHouse 挂起
        clean_codes = set()
        for file_name in csv_files:
            raw_code = file_name.split("-")[0]
            clean_code = raw_code.upper()
            for jq_suffix, official_suffix in EXCHANGE_SUFFIX_MAP.items():
                clean_code = clean_code.replace(jq_suffix, official_suffix)
            clean_codes.add(clean_code)

        if clean_codes:
            code_list_str = ", ".join(f"'{c}'" for c in sorted(clean_codes))
            print(f"正在批量清理现有数据，涉及合约数: {len(clean_codes)}...")
            client.command(
                f"ALTER TABLE default.futures_1m_bar DELETE WHERE code IN ({code_list_str})"
            )
            time.sleep(1.0)  # 给 ClickHouse 后台 Mutation 处理让出启动时间

        for idx, file_name in enumerate(csv_files, start=1):
            # 3.2 提取文件名中的合约代码并标准化
            raw_code = file_name.split("-")[0]
            clean_code = raw_code.upper()
            for jq_suffix, official_suffix in EXCHANGE_SUFFIX_MAP.items():
                clean_code = clean_code.replace(jq_suffix, official_suffix)

            # 3.3 读取数据
            data_bytes = zf.read(file_name)
            df_raw = pl.read_csv(io.BytesIO(data_bytes))

            if df_raw.is_empty():
                print(
                    f"[{idx}/{len(csv_files)}] 合约 {clean_code} 无分钟线数据，跳过。"
                )
                continue

            # 3.4 数据清洗转换
            df_cleaned = clean_and_normalize_data(df_raw, clean_code)

            # 3.5 写入 ClickHouse (调用防网关超时且支持指数退避重试的写入函数)
            safe_insert_df(client, "default.futures_1m_bar", df_cleaned)

            rows_count = len(df_cleaned)
            total_rows += rows_count
            print(
                f"[{idx}/{len(csv_files)}] 成功导入 {clean_code} | 行数: {rows_count}"
            )
            time.sleep(0.05)  # 每次微休眠以防 ClickHouse 服务器瞬时过载

    elapsed = time.time() - start_time
    print(f"完成 ZIP {zip_path.name} | 耗时: {elapsed:.2f} 秒 | 导入行数: {total_rows}")
    return total_rows


def main() -> None:
    args = parse_args()

    # 1. 尝试建立 ClickHouse 连接，优先尝试内网 IP，超时则降级到配置文件中的域名
    db_config = load_db_config(args.config_path)
    local_host = "192.168.100.99"
    local_port = 8123
    client = None

    print(f"正在尝试连接内网 ClickHouse 实例 (Host: {local_host}, Port: {local_port})...")
    try:
        client = clickhouse_connect.get_client(
            host=local_host,
            port=local_port,
            username=db_config["username"],
            password=db_config["password"],
            database=db_config["database"],
            secure=False,
            connect_timeout=3,  # 3秒超时，避免在非局域网环境下卡挂过久
        )
        print(f"内网极速通道连接成功！(数据库版本: {client.command('SELECT version()')})")
    except Exception as e:
        print(f"内网通道连接失败或超时: {e}")
        secure = db_config.get("port") == 443 or db_config.get("host") == "ch.jieyujing.eu.org"
        fallback_host = db_config.get("host")
        fallback_port = db_config.get("port")
        print(
            f"正在自动切换至备用公网通道... (Host: {fallback_host}, Port: {fallback_port}, Secure: {secure})"
        )
        try:
            client = clickhouse_connect.get_client(
                host=fallback_host,
                port=fallback_port,
                username=db_config["username"],
                password=db_config["password"],
                database=db_config["database"],
                secure=secure,
            )
            print("公网备用通道连接成功！")
        except Exception as fallback_err:
            print(f"内网与公网通道均连接失败，请检查数据库配置与网络: {fallback_err}")
            raise fallback_err

    # 2. 校验/创建大表
    init_clickhouse_table(client)

    if args.refresh_only:
        refresh_data_index(client)
        return

    input_path = Path(args.zip_path)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到指定的输入路径: {input_path}")

    # 3. 决定处理是单文件还是目录
    start_time = time.time()
    total_imported_rows = 0

    if input_path.is_dir():
        print(f"输入路径为目录，正在扫描: {input_path}")
        zip_files = sorted(input_path.glob("*.zip"))
        print(f"检测到 {len(zip_files)} 个待导入的 ZIP 压缩包。")
        for idx, zip_path in enumerate(zip_files, start=1):
            print(f"\n[正在导入品种组 {idx}/{len(zip_files)}]")
            total_imported_rows += import_single_zip(client, zip_path, args.limit_files)
    else:
        total_imported_rows += import_single_zip(client, input_path, args.limit_files)

    elapsed = time.time() - start_time
    print("\n==========================================")
    print("【全部导入完毕】")
    print(f"总计耗时: {elapsed:.2f} 秒")
    print(f"累计导入总行数: {total_imported_rows} 行")
    print("==========================================")

    # 4. 自动刷新数据索引表
    refresh_data_index(client)


if __name__ == "__main__":
    main()
