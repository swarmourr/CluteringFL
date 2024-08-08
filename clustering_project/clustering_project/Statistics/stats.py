import pandas as pd
import numpy as np

class DataFrameStatistics:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Parameters:
        - data: pd.DataFrame
            The input DataFrame for which to compute statistics.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        self.data = data
        self.clients_stats_data = pd.DataFrame()


    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for a DataFrame.

        Parameters:
        - df: pd.DataFrame
            The DataFrame for which to compute statistics.

        Returns:
        - pd.DataFrame
            A DataFrame containing statistics for the input DataFrame.
        """
        df_numeric = df.select_dtypes(include=[np.number])
        
        # Calculate statistics
        stats = pd.DataFrame({
            'mean': df_numeric.mean(),
            'std': df_numeric.std(),
            'min': df_numeric.min(),
            '25%': df_numeric.quantile(0.25),
            '50%': df_numeric.median(),
            '75%': df_numeric.quantile(0.75),
            'max': df_numeric.max(),
            'median': df_numeric.median(),
            'variance': df_numeric.var(),
            'skew': df_numeric.skew(),
            'kurtosis': df_numeric.kurtosis(),
        })
        
        return stats

    def feature_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each feature in the DataFrame.

        Returns:
        - pd.DataFrame
            A DataFrame containing statistics for each feature.
        """
        feature_stats = self.compute_statistics(self.data)
        
        # Reshape the DataFrame to have one row per feature and one column per statistic
        feature_stats_flat = feature_stats.reset_index()
        feature_stats_flat = pd.melt(feature_stats_flat, id_vars='index', var_name='Statistic', value_name='Value')
        feature_stats_flat = feature_stats_flat.pivot_table(index='index', columns='Statistic', values='Value')
        feature_stats_flat.index.name = 'Feature'
        feature_stats_flat.columns.name = None  # Remove the name of the columns axis
        
        return feature_stats_flat

    def dataset_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for the entire DataFrame, aggregating feature-wise statistics.

        Returns:
        - pd.DataFrame
            A DataFrame containing overall statistics for the dataset.
        """
        stats = self.compute_statistics(self.data)
        
        # Aggregate statistics across features
        dataset_stats = {
            'mean': [stats['mean'].mean()],
            'std': [stats['std'].mean()],
            'min': [stats['min'].min()],
            '25%': [stats['25%'].mean()],
            '50%': [stats['50%'].mean()],
            '75%': [stats['75%'].mean()],
            'max': [stats['max'].max()],
            'median': [stats['50%'].mean()],
            'variance': [stats['variance'].mean()],
            'skew': [stats['skew'].mean()],
            'kurtosis': [stats['kurtosis'].mean()],
        }
        
        dataset_stats_df = pd.DataFrame(dataset_stats, index=['dataset'])
        dataset_stats_df.index.name = 'Feature'
        return dataset_stats_df

    def all_statistics(self) -> pd.DataFrame:
        """
        Combine feature-wise and dataset-wise statistics into one DataFrame.

        Returns:
        - pd.DataFrame
            A DataFrame containing both feature-wise and dataset-wise statistics.
        """
        feature_stats = self.feature_statistics()
        dataset_stats = self.dataset_statistics()
        
        # Combine feature-wise statistics with dataset statistics
        all_stats = pd.concat([feature_stats, dataset_stats], axis=0)
        return all_stats
    
    def create_feature_stat_df(self,dataset_stats_df) -> pd.DataFrame:
            """
            Create a single-row DataFrame with columns in the format 'feature_statistic'.

            Returns:
            pd.DataFrame: A single-row DataFrame with columns in the format 'feature_statistic'.
            """
            

            # Transpose the DataFrame and flatten it
            flattened_df = dataset_stats_df.T.transpose().stack()

            # Create new column names by combining row and column indices
            flattened_df.index = [f"{col}_{idx}" for idx, col in flattened_df.index]

            # Convert to a single row DataFrame
            single_row_df = flattened_df.to_frame().transpose()

            
            return single_row_df
    
   
            

