# -------------------------------------------------------
# Week 4 A2 - Data Aggregation using World Happiness Dataset
# Author : Benjelyn Reves Patiag
# Date   : 6 May 2026
# Description:
#     It run two SQL aggregation query using SQLite.
#     First query is about GDP category and happiness ranking.
#     Second query is about corruption perception group comparison.
# -------------------------------------------------------


# ─────────────────────────────────────────────
# Import the library that we need for this work
# ─────────────────────────────────────────────
import sqlite3                      #  SQLite database
import textwrap                     # clean up the SQL string indentation
from dataclasses import dataclass, field  # dataclass is nice way to hold setting value
from pathlib import Path            # Path is better than string for file location
from typing import Dict, Optional   # for type hint, make code more readable

import matplotlib.pyplot as plt     # we use this for drawing the table as picture
import matplotlib.colors as mcolors # this is for nice colour on table header
import pandas as pd                 # pandas help us read and work with data table


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION DATACLASS
#    We put all setting in one place so easy to change later
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    This class is holding all the path and setting for the whole script.
    If you want to change file name or folder, just change here only,
    no need to go hunt every line in the code.
    """

    # --- where the files are living ---
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # --- the CSV file that has the happiness data ---
    csv_file: str = "world_happiness_dataset.csv"

    # --- name for the SQLite database file we going to create ---
    db_file: str = "world_happiness.db"


    # --- folder where we save the output picture and CSV ---
    output_dir: str = "outputs"

    # --- GDP boundary number for categorise country ---
    gdp_low_threshold: float = 0.80    # below this = Low GDP
    gdp_high_threshold: float = 1.20   # above or equal this = High GDP

    # --- colour for table header when we draw the picture ---
    header_color: str = "#1a5276"      # dark blue colour
    row_even_color: str = "#d6eaf8"    # light blue for even row
    row_odd_color: str = "#ffffff"     # white for odd row

    # --- resolution for saving picture ---
    image_dpi: int = 200

    @property
    def csv_path(self) -> Path:
        """Full path to the CSV file."""
        return self.base_dir / self.csv_file

    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database file."""
        return self.base_dir / self.db_file



    @property
    def output_path(self) -> Path:
        """Full path to the output folder."""
        return self.base_dir / self.output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SQL QUERY DEFINITIONS
#    We keep all the SQL in one class so easy to find and maintain
# ═══════════════════════════════════════════════════════════════════════════════

class WorldHappinessQueries:
    """
    This class is hold the two SQL query that we need to run.
    We define them as class-level attribute so any part of script can access.

    Query 1: Group country by GDP category, calculate average happiness,
             and rank country inside each group.

    Query 2: Split country into high and low corruption perception group,
             then calculate many average and compare using subquery.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # QUERY 1: GDP CATEGORY, AVERAGE HAPPINESS, AND COUNTRY RANKING
    #
    # This query is doing three thing in sequence using CTE (Common Table Expression):
    #
    # Step A (gdp_grouped):
    #   We label every country with GDP category using CASE statement.
    #   Country with GDP_per_Capita below 0.80 get label 'Low GDP'.
    #   Country from 0.80 to below 1.20 get label 'Medium GDP'.
    #   Country 1.20 and above get label 'High GDP'.
    #   This create new column called GDP_Category for every row.
    #
    # Step B (category_average):
    #   We group by GDP_Category and calculate the average happiness score
    #   for every group using AVG() function.
    #   This give us one row per GDP category with the average happiness.
    #
    # Step C (ranked_countries):
    #   We use RANK() window function with PARTITION BY GDP_Category.
    #   This mean ranking restart fresh for each category.
    #   Country with highest happiness score inside category get rank 1.
    #
    # Final SELECT:
    #   We join country-level data with category average using GDP_Category.
    #   Then order result by GDP category order and then by rank inside category.
    # ──────────────────────────────────────────────────────────────────────────
    QUERY_1: str = textwrap.dedent("""
        -- =========================================================
        -- QUERY 1: GDP Category, Average Happiness, Country Ranking
        -- =========================================================

        -- Step A: Label every country with GDP category using CASE
        WITH gdp_grouped AS (
            SELECT
                Country,
                Happiness_Score,
                GDP_per_Capita,
                CASE
                    WHEN GDP_per_Capita < 0.80  THEN 'Low GDP'
                    WHEN GDP_per_Capita >= 0.80
                     AND GDP_per_Capita <  1.20  THEN 'Medium GDP'
                    ELSE                              'High GDP'
                END AS GDP_Category
            FROM world_happiness
        ),

        -- Step B: Calculate average happiness for each GDP category
        category_average AS (
            SELECT
                GDP_Category,
                ROUND(AVG(Happiness_Score), 2) AS Average_Happiness
            FROM gdp_grouped
            GROUP BY GDP_Category
        ),

        -- Step C: Rank countries inside each GDP category by happiness score
        ranked_countries AS (
            SELECT
                Country,
                GDP_Category,
                GDP_per_Capita,
                Happiness_Score,
                RANK() OVER (
                    PARTITION BY GDP_Category      -- ranking restart for each category
                    ORDER BY Happiness_Score DESC  -- highest happiness get rank 1
                ) AS Happiness_Rank
            FROM gdp_grouped
        )

        -- Final: Join rank with category average, then order nicely
        SELECT
            r.GDP_Category,
            r.Country,
            r.GDP_per_Capita,
            r.Happiness_Score,
            a.Average_Happiness,
            r.Happiness_Rank
        FROM ranked_countries r
        JOIN category_average a
            ON r.GDP_Category = a.GDP_Category
        ORDER BY
            -- Manual order: Low=1, Medium=2, High=3
            CASE r.GDP_Category
                WHEN 'Low GDP'    THEN 1
                WHEN 'Medium GDP' THEN 2
                WHEN 'High GDP'   THEN 3
            END,
            r.Happiness_Rank;
    """).strip()

    # ──────────────────────────────────────────────────────────────────────────
    # QUERY 2: HIGH VS LOW CORRUPTION PERCEPTION COMPARISON
    #
    # This query is doing comparison between two group of country
    # based on how they perceive corruption, using subquery as benchmark.
    #
    # Step A (corruption_groups CTE):
    #   First we calculate overall average corruption perception using subquery
    #   inside the CASE statement.  This is scalar subquery — return one number.
    #   Country with corruption perception >= that average → 'High Corruption Perception'
    #   Country with corruption perception <  that average → 'Low Corruption Perception'
    #
    # Step B (main SELECT with GROUP BY):
    #   Group the result by Corruption_Group.
    #   For each group we calculate:
    #     - COUNT(*): how many country in the group
    #     - AVG(Happiness_Score): average happiness
    #     - AVG(GDP_per_Capita): average GDP
    #     - AVG(Social_Support): average social support
    #     - AVG(Healthy_Life_Expectancy): average life expectancy
    #     - AVG(Freedom_to_Make_Choices): average freedom score
    #     - AVG(Generosity): average generosity
    #     - AVG(Perceptions_of_Corruption): average corruption perception
    #
    # Step C (Difference_From_Overall_Happiness):
    #   Another subquery inside SELECT calculate overall happiness average.
    #   We subtract that from group average to see how far above or below.
    #   Positive number mean group is happier than overall average.
    #   Negative number mean group is less happy than overall average.
    # ──────────────────────────────────────────────────────────────────────────
    QUERY_2: str = textwrap.dedent("""
        -- =========================================================
        -- QUERY 2: High vs Low Corruption Perception Comparison
        -- =========================================================

        -- Step A: Label every country as high or low corruption perception
        --         using subquery to get the overall average as benchmark
        WITH corruption_groups AS (
            SELECT
                Country,
                Happiness_Score,
                GDP_per_Capita,
                Social_Support,
                Healthy_Life_Expectancy,
                Freedom_to_Make_Choices,
                Generosity,
                Perceptions_of_Corruption,
                CASE
                    WHEN Perceptions_of_Corruption >= (
                        -- Scalar subquery: this return one number - the overall average
                        SELECT AVG(Perceptions_of_Corruption)
                        FROM world_happiness
                    ) THEN 'High Corruption Perception'
                    ELSE   'Low Corruption Perception'
                END AS Corruption_Group
            FROM world_happiness
        )

        -- Step B & C: Group by corruption group, calculate multiple averages,
        --             and compare each group to overall happiness average
        SELECT
            Corruption_Group,
            COUNT(*)                                            AS Country_Count,
            ROUND(AVG(Happiness_Score),          2)            AS Avg_Happiness,
            ROUND(AVG(GDP_per_Capita),           2)            AS Avg_GDP,
            ROUND(AVG(Social_Support),           2)            AS Avg_Social_Support,
            ROUND(AVG(Healthy_Life_Expectancy),  2)            AS Avg_Life_Expectancy,
            ROUND(AVG(Freedom_to_Make_Choices),  2)            AS Avg_Freedom,
            ROUND(AVG(Generosity),               2)            AS Avg_Generosity,
            ROUND(AVG(Perceptions_of_Corruption),2)            AS Avg_Corruption_Perception,

            -- Step C: Subtract overall happiness average using another subquery
            ROUND(
                AVG(Happiness_Score) - (
                    SELECT AVG(Happiness_Score)
                    FROM world_happiness
                ), 2
            )                                                   AS Difference_From_Overall_Happiness
        FROM corruption_groups
        GROUP BY Corruption_Group
        ORDER BY Avg_Happiness DESC;
    """).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATABASE MANAGER CLASS
#    This class is responsible for all database operation
#    (load data, run query, close connection)
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    This class is managing the SQLite database.
    It know how to load CSV into database, run SQL query,
    and return the result as pandas DataFrame.

    We use context manager pattern (with statement) internally
    so connection always close properly even if error happen.
    """

    def __init__(self, db_path: Path) -> None:
        # save the database path so we can use later
        self.db_path = db_path
        print(f"[DatabaseManager] Database path is set to: {self.db_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Method: load CSV file into SQLite table
    # ──────────────────────────────────────────────────────────────────────────
    def load_csv_to_table(self, csv_path: Path, table_name: str = "world_happiness") -> pd.DataFrame:
        """
        Read the CSV file using pandas, then push into SQLite as a table.
        If table already exist, we replace it so data is always fresh.

        Args:
            csv_path: where the CSV file is sitting
            table_name: what name we give the table in SQLite

        Returns:
            The DataFrame that was loaded (useful for checking)
        """
        # first we load the CSV into pandas DataFrame
        # this is like putting the spreadsheet into memory
        print(f"[DatabaseManager] Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)

        # sometimes column name have extra space at front or back
        # we strip that away so SQL query will not break
        df.columns = [col.strip() for col in df.columns]

        print(f"[DatabaseManager] CSV loaded — {len(df)} rows, {len(df.columns)} columns")
        print(f"[DatabaseManager] Column names: {list(df.columns)}")

        # now we connect to SQLite and push the DataFrame as a table
        # if_exists='replace' mean we delete old table first, then create new
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"[DatabaseManager] Table '{table_name}' is created/replaced in database")

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Method: run a SQL query and get result as DataFrame
    # ──────────────────────────────────────────────────────────────────────────
    def run_query(self, sql: str) -> pd.DataFrame:
        """
        Execute one SQL query against the database and return result as DataFrame.
        We use read_sql_query from pandas because it automatically
        put column name and convert data type nicely.

        Args:
            sql: the SQL string we want to execute

        Returns:
            DataFrame containing all the rows from query result
        """
        # connect to database and run the query
        # pandas read_sql_query handle everything — column name, data type, etc.
        with sqlite3.connect(self.db_path) as conn:
            result_df = pd.read_sql_query(sql, conn)

        print(f"[DatabaseManager] Query returned {len(result_df)} rows")
        return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RESULT EXPORTER CLASS
#    This class is responsible for saving the result to file
#    (CSV files and PNG screenshot-style image)
# ═══════════════════════════════════════════════════════════════════════════════

class ResultExporter:
    """
    This class is doing all the export work.
    It can save DataFrame as CSV file and also as nice-looking PNG image.
    We design it to be reusable — same exporter can handle any query result.
    """

    def __init__(self, output_dir: Path, config: Config) -> None:
        # save the output directory path
        self.output_dir = output_dir

        # also save the config so we can use the colour setting
        self.config = config

        # make sure the output folder is exist, create if not
        # parents=True mean create parent folder too if needed
        # exist_ok=True mean no error if folder already there
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ResultExporter] Output directory ready at: {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Method: save DataFrame to CSV file
    # ──────────────────────────────────────────────────────────────────────────
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save the DataFrame result as CSV file.
        index=False mean we don't want the row number column in the file.

        Args:
            df: the DataFrame we want to save
            filename: what to call the file (just the name, not full path)

        Returns:
            The full path where the file was saved
        """
        # build the full output path by joining folder + filename
        output_path = self.output_dir / filename

        # save without row index because index number is not useful in output
        df.to_csv(output_path, index=False)
        print(f"[ResultExporter] CSV saved: {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # Method: save DataFrame as screenshot-style PNG image
    # ──────────────────────────────────────────────────────────────────────────
    def save_table_image(self, df: pd.DataFrame, title: str, filename: str) -> Path:
        """
        Draw the DataFrame as a nice table picture and save as PNG.
        We calculate the figure size based on how many rows and column the data have
        so the table always fit properly in the image.

        Args:
            df: the DataFrame we want to draw as table
            title: text to show at the top of the image
            filename: what to call the PNG file

        Returns:
            The full path where the image was saved
        """
        # calculate how big the figure should be based on data size
        # more column = wider figure, more row = taller figure
        num_rows, num_cols = df.shape
        fig_width  = max(12, num_cols * 1.6)   # at least 12 wide, more if many column
        fig_height = max(4,  num_rows * 0.42 + 1.6)  # at least 4 tall, more if many row

        # create the matplotlib figure and axis (axis = drawing area)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # turn off the normal axis (no x-axis line, no y-axis line, no tick)
        # we only want to see the table, not the axis
        ax.axis("off")

        # set the title at top of image
        ax.set_title(title, fontsize=13, fontweight="bold", pad=16,
                     color="#1a1a1a")

        # create the table inside the axis
        # cellText = the actual data value
        # colLabels = the column header name
        the_table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )

        # set font size for all cell in table
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8.5)

        # scale the table: 1.0 horizontal, 1.4 vertical (make row a bit taller)
        the_table.scale(1, 1.4)

        # now we apply nice colour to each cell
        for (row_idx, col_idx), cell in the_table.get_celld().items():

            # remove the default border from cell
            cell.set_edgecolor("#cccccc")

            if row_idx == 0:
                # this is the header row — give it dark blue background and white text
                cell.set_facecolor(self.config.header_color)
                cell.set_text_props(color="white", weight="bold", fontsize=9)
            else:
                # for data row, alternate between light blue and white
                # this make it easier to read across the row
                if row_idx % 2 == 0:
                    cell.set_facecolor(self.config.row_even_color)
                else:
                    cell.set_facecolor(self.config.row_odd_color)

                # make text a bit smaller in data row
                cell.set_text_props(fontsize=8)

        # auto-adjust column width based on content
        the_table.auto_set_column_width(list(range(num_cols)))

        # build output path
        output_path = self.output_dir / filename

        # save the figure with tight bounding box so nothing get cut off
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.image_dpi, bbox_inches="tight",
                    facecolor="white")  # white background for the image
        plt.close(fig)  # close figure to free memory

        print(f"[ResultExporter] Image saved: {output_path}")
        return output_path



# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRINTER / REPORTER CLASS
#    This class is for printing summary and result to the terminal
#    in a nice, formatted way
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolePrinter:
    """
    This class is handling all the print output to terminal.
    We separate printing logic from business logic so code is cleaner.
    """

    @staticmethod
    def print_header(title: str) -> None:
        """Print a big header in terminal so easy to see where section start."""
        # calculate width based on title length, minimum 60 character
        width = max(60, len(title) + 6)
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)

    @staticmethod
    def print_dataframe(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
        """
        Print DataFrame nicely to terminal.
        If max_rows is given, only show that many row.
        """
        # set pandas option so column not get cut off or wrap
        with pd.option_context(
            "display.max_columns", None,      # show all column
            "display.max_colwidth", 30,        # but limit each column width
            "display.width", 200,              # wide terminal output
            "display.float_format", "{:.2f}".format,  # 2 decimal place for float
        ):
            if max_rows:
                print(df.head(max_rows).to_string(index=False))
                if len(df) > max_rows:
                    print(f"  ... and {len(df) - max_rows} more rows (see CSV for full result)")
            else:
                print(df.to_string(index=False))

    @staticmethod
    def print_summary(label: str, df: pd.DataFrame) -> None:
        """Print a short summary about the DataFrame — row count and column names."""
        print(f"\n[Summary] {label}")
        print(f"  Total rows    : {len(df)}")
        print(f"  Total columns : {len(df.columns)}")
        print(f"  Columns       : {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN ANALYSIS ORCHESTRATOR CLASS
#    This is the "boss" class that coordinate all the other class.
#    It call DatabaseManager to load data and run query,
#    then call ResultExporter to save the output,
#    then call ConsolePrinter to show result in terminal.
# ═══════════════════════════════════════════════════════════════════════════════

class WorldHappinessAnalysis:
    """
    This is the main orchestrator class for Part 2 analysis.
    It bring together all the smaller class and run the full analysis
    in the correct order.

    The flow is:
        1. Setup config and create output folder
        2. Load CSV data into SQLite database
        3. Run Query 1 (GDP categories + ranking)
        4. Run Query 2 (Corruption perception comparison)
        5. Export both result as CSV and PNG
        6. Print result to terminal

    Usage:
        analysis = WorldHappinessAnalysis()
        analysis.run()
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialise the analysis with configuration.
        If no config given, we use the default Config setting.
        """
        # if no config given, create default one
        self.config = config or Config()

        # print the config so user know what setting is being used
        print("[WorldHappinessAnalysis] Initialising with config:")
        print(f"  CSV file   : {self.config.csv_path}")
        print(f"  Database   : {self.config.db_path}")

        print(f"  Output dir : {self.config.output_path}")

        # create the helper objects (this is "composition" in OOP)
        # each object has one job and we combine them in this main class
        self.db_manager   = DatabaseManager(self.config.db_path)
        self.exporter     = ResultExporter(self.config.output_path, self.config)
        self.printer      = ConsolePrinter()
        self.queries      = WorldHappinessQueries()

        # placeholder for the query result, we fill these during run()
        self.result_query1: Optional[pd.DataFrame] = None
        self.result_query2: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Load CSV into database
    # ──────────────────────────────────────────────────────────────────────────
    def _step_load_data(self) -> None:
        """
        Load the CSV happiness data into SQLite.
        This is the first thing we must do before running any query.
        """
        self.printer.print_header("STEP 1: Load CSV Data into SQLite Database")

        # this line call the DatabaseManager to do the loading work
        # the DataFrame is returned but we only keep it for summary print
        loaded_df = self.db_manager.load_csv_to_table(
            csv_path=self.config.csv_path,
            table_name="world_happiness",
        )

        # print summary about what was loaded
        self.printer.print_summary("World Happiness Data Loaded", loaded_df)


    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Run Query 1 (GDP Categories)
    # ──────────────────────────────────────────────────────────────────────────
    def _step_run_query1(self) -> None:
        """
        Execute Query 1 which is grouping country by GDP category,
        calculating average happiness per category, and ranking
        country inside each category by happiness score.
        """
        self.printer.print_header("STEP 2: Run Query 1: GDP Categories, Avg Happiness, Ranking")

        # run the SQL and store result in self.result_query1
        self.result_query1 = self.db_manager.run_query(self.queries.QUERY_1)

        # print what we got in terminal (show first 20 rows to not flood screen)
        print("\nQuery 1 Result Preview (first 20 rows):")
        self.printer.print_dataframe(self.result_query1, max_rows=20)
        self.printer.print_summary("Query 1 Full Result", self.result_query1)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Run Query 2 (Corruption Perception)
    # ──────────────────────────────────────────────────────────────────────────
    def _step_run_query2(self) -> None:
        """
        Execute Query 2 which is splitting country into high and low
        corruption perception group, then calculating multiple averages
        and comparing each group to overall average using subquery.
        """
        self.printer.print_header("STEP 3: Run Query 2: Corruption Perception Group Comparison")

        # run the SQL and store result in self.result_query2
        self.result_query2 = self.db_manager.run_query(self.queries.QUERY_2)

        # print the full result (only 2 rows so show all)
        print("\nQuery 2 Full Result:")
        self.printer.print_dataframe(self.result_query2)
        self.printer.print_summary("Query 2 Full Result", self.result_query2)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Export all result to files
    # ──────────────────────────────────────────────────────────────────────────
    def _step_export_results(self) -> None:
        """
        Save the query result to CSV file and PNG image.
        We save both format so the result is useful for report and analysis.
        """
        self.printer.print_header("STEP 4: Export Results to CSV and PNG Files")

        # export Query 1 result as CSV
        self.exporter.save_csv(
            df=self.result_query1,
            filename="query1_gdp_category_ranking_results.csv",
        )

        # export Query 1 result as PNG table image
        self.exporter.save_table_image(
            df=self.result_query1,
            title="Query 1 Result: GDP Categories, Average Happiness, and Country Ranking",
            filename="query1_gdp_category_ranking_screenshot.png",
        )

        # export Query 2 result as CSV
        self.exporter.save_csv(
            df=self.result_query2,
            filename="query2_corruption_group_comparison_results.csv",
        )

        # export Query 2 result as PNG table image
        self.exporter.save_table_image(
            df=self.result_query2,
            title="Query 2 Result: Corruption Perception Group Comparison",
            filename="query2_corruption_group_comparison_screenshot.png",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Print final findings summary
    # ──────────────────────────────────────────────────────────────────────────
    def _step_print_findings(self) -> None:
        """
        Print a human-readable summary of the key findings from both query.
        This is not the raw data — this is the interpretation.
        """
        self.printer.print_header("STEP 5: Key Findings Summary")

        # ── Query 1 findings ──
        print("\n[Query 1: GDP Category Analysis]")

        # group the result by GDP_Category and show average happiness per group
        if self.result_query1 is not None:
            gdp_summary = (
                self.result_query1
                .drop_duplicates(subset="GDP_Category")   # one row per category
                [["GDP_Category", "Average_Happiness"]]
                .set_index("GDP_Category")
            )
            for category, row in gdp_summary.iterrows():
                print(f"  {category:<14}: Average Happiness = {row['Average_Happiness']:.2f}")

            # also show the top ranked country per GDP category
            top_per_category = (
                self.result_query1[self.result_query1["Happiness_Rank"] == 1]
                [["GDP_Category", "Country", "Happiness_Score"]]
            )
            print("\n  Rank #1 Country per GDP Category:")
            for _, row in top_per_category.iterrows():
                print(f"  {row['GDP_Category']:<14}: {row['Country']}"
                      f" (Score: {row['Happiness_Score']:.2f})")

        # ── Query 2 findings ──
        print("\n[Query 2: Corruption Perception Analysis]")
        if self.result_query2 is not None:
            for _, row in self.result_query2.iterrows():
                diff_sign = "+" if row["Difference_From_Overall_Happiness"] >= 0 else ""
                print(f"  {row['Corruption_Group']:<30}: "
                      f"Avg Happiness = {row['Avg_Happiness']:.2f} "
                      f"(Diff from overall: {diff_sign}{row['Difference_From_Overall_Happiness']:.2f})")

        print("\n  Interpretation:")
        print("  - Countries with LOW corruption perception tend to have HIGHER happiness.")
        print("  - Countries with HIGH corruption perception tend to have LOWER happiness.")
        print("  - GDP also play important role - richer countries generally score higher.")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD: run the full analysis from start to finish
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        """
        Run the full analysis pipeline from loading data to exporting result.
        This is the only public method you need to call from outside.
        All the step method are private (start with underscore) and called here.
        """
        print("\n" + "█" * 65)
        print("█  Part 2: World Happiness SQL Aggregation Analysis       █")
        print("█  Author: Benjelyn Reves Patiag | MSE803 Data Analytics   █")
        print("█" * 65)

        # run each step in order — if one fail, everything stop
        self._step_load_data()
        self._step_run_query1()
        self._step_run_query2()
        self._step_export_results()
        self._step_print_findings()

        # final success message
        print("=" * 65)
        print(f"  All done! Output files are in: {self.config.output_path}")
        print("=" * 65 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
#    This block only run when we execute this file directly (not when imported).
#    Standard Python convention: if __name__ == "__main__"
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # create the analysis object using default config
    # if you want to change setting, pass a custom Config() here
    analysis = WorldHappinessAnalysis()

    # run everything from start to finish
    analysis.run()
