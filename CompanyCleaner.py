import pandas as pd
import string
import numpy as np
import re
from tqdm import tqdm
from collections import Counter
from functools import partial
from typing import List
import pdb
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class CompanyCleaner:
    def __init__(self, dataframe: pd.DataFrame, column_name: str) -> None:
        """Initialize the CompanyCleaner object with a pandas DataFrame and a column name containing company names."""
        self.df = dataframe#[0:1_000].copy()  # DataFrame containing the data
        # self.df.drop_duplicates([column_name], inplace=True)
        # self.df = self.df.sample(n=100_000, random_state=42)
        self.column_name = column_name  # Column name containing company names
        self.cleaned_column_name = f"{column_name}_cleaned"  # Name for the cleaned column

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from a string and replace it with whitespace to avoid joining words together."""
        """Some punctuation should be joined, while others should not, need to fix """
        text = re.sub('\.|\,', '', text)
        return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in a string by replacing multiple spaces with a single space."""
        return ' '.join(text.split())

    def _perform_initial_cleaning(self) -> None:
        """Perform initial cleaning steps on the company names."""
        # Replace '&' with 'AND' for consistency
        self.df[self.cleaned_column_name] = self.df[self.column_name].str.replace('&', ' AND ')
        # Remove punctuation and normalize whitespace
        self.df[self.cleaned_column_name] = self.df[self.cleaned_column_name].apply(
            lambda x: self._normalize_whitespace(self._remove_punctuation(x))
        )

    def _get_common_endings_and_remove(self, ending_count, occurrence_count) -> pd.DataFrame:
        """Identify common endings in company names based on a minimum occurrence count."""
        # Create columns that will be 
        # self.df[self.cleaned_column_name + '_4' ] = self.df[self.cleaned_column_name].copy() # fix this
        # self.df[self.cleaned_column_name + '_4' ] = None # fix this

        def get_most_common_endings(count, old_column_name):
            # Extract the last word from each company name
            endings = self.df[old_column_name].str.split().str[-count:]
            endings = [' '.join(ending) for ending in endings]
            # Count occurrences of each ending
            ending_counts = Counter(endings)
            # Filter endings by minimum count # use a frequency % or some other statistical count for this
            common_endings = [ending for ending, count in ending_counts.items() if count >= occurrence_count]
            return common_endings

        def create_exception_0_index():
            '''
            Compile index of entities that that have already had endings extracted
            '''
            masked_exception = self.df['Difference'].notnull()
            idx_exceptions_0 = self.df[masked_exception].index.tolist()
            return idx_exceptions_0

        def create_exception_1_index(ending_count):#:
            '''
            Compile index of entities that have less words than the count of the ending being extracted
            ex: ending count = 3, 'TRADER LIMITED LLC' is an exempted entity 
            '''
            return []
            # if ending_count == 1:
            #     pdb.set_trace()
            # Exception 1 don't alter strings that have the same number of words as common_ending
            # masked_exception = self.df[self.cleaned_column_name].str.split().str.len() <= ending_count
            # idx_exceptions_1 = self.df[masked_exception].index.tolist()
            # return idx_exceptions_1

        def create_exception_2_index(count, old_column_name):
            '''
            If string contains a preposition, and the number of words AFTER the preposition is less than
            the ending count, it is exempted:
            e: if count = 2, 'TRADED AND FISH LLC' is exempted because 2 words after 'AND', 2 = 2
            e: if count = 1, 'TRADED AND FISH LLC' is NOT exempted because 2 words after 'AND', 2 > 1
            the number of words after the preposition must be less than the count
            '''
            prepositions = {'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'about', 'as', 'into', 
                            'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 
                            'without', 'before', 'under', 'around', 'among', 'and'}
            def count_words_after_last_preposition(text):
                words = text.split()
                for i in range(len(words)-1, -1, -1):  # Iterate backwards
                    if words[i].lower() in prepositions:
                        return len(words) - i - 1  # Number of words after the last preposition
                return None  # In case there's no preposition

            self.df['words_after_preposition'] = self.df[old_column_name].apply(count_words_after_last_preposition)
            idx_exceptions_2 = self.df[(self.df['words_after_preposition'] <= count)].index.tolist()
            self.df = self.df.drop(columns=['words_after_preposition'])
            return idx_exceptions_2

        def create_exception_3_index(count, old_column_name, string_length=10):
            '''
            So this is looking at the length of the remaining string if the ending is removed.
            If the remaining string is length than the 'string_length' variable, then it's untouched. 
            '''
            # if count == 3:
            #     pdb.set_trace()
            self.df['string_length'] = self.df[old_column_name].apply(lambda x: len(' '.join(x.split()[:-count]) if len(x.split()) > count else ''))
            masked_exception = self.df['string_length'] < string_length
            idx_exceptions_3 = self.df[masked_exception].index.tolist()
            return idx_exceptions_3

        def remove_common_endings(common_endings, idx_exceptions, old_column_name, new_column_name):
            '''
            Take the common endings and remove them from values that are not idx_exceptions. 
            Save a new column for the values that have had the endings removed.
            Returns a dataframe with the new "cleaned" column
            '''
            # Create a regex pattern to match common endings
            regex_pattern = r'\b' + r'$|\b'.join(common_endings) + r'$'
            # Remove common endings from company names if they are not in the exceptions
            self.df.loc[idx_exceptions, new_column_name] = self.df.loc[idx_exceptions, old_column_name].str.replace(regex_pattern, '', regex=True).str.strip()
            # Replace NaN values in the new column with the original company names
            self.df[new_column_name] = np.where(self.df[new_column_name].isnull(), self.df[old_column_name], self.df[new_column_name])

        def populate_column_for_removed_endings(idx_exceptions, count, old_column_name, new_column_name):
            '''
            Create a column to store the endings that were removed from the company names
            '''
            
            self.df[f'{count}_word_ending_removed'] = None
            zipped_generator = zip(self.df.loc[idx_exceptions, new_column_name], self.df.loc[idx_exceptions, old_column_name])
            self.df.loc[idx_exceptions, f'{count}_word_ending_removed'] = [j.replace(i, '').strip('()') for i, j in zipped_generator]
            self.df[f'{count}_word_ending_removed']  = self.df[f'{count}_word_ending_removed'].replace('', None)

        for count in range(ending_count, 0, -1):
            if count == ending_count:
                old_column_name = self.cleaned_column_name
                new_column_name = f'{self.cleaned_column_name}_{count}'
            else:
                old_column_name = f'{self.cleaned_column_name}_{count + 1}'
                new_column_name = f'{self.cleaned_column_name}_{count}'
            
            logging.info(f'Beginning extraction for most common {count}-word endings')
            common_endings = get_most_common_endings(count, old_column_name)
            logging.info('Word Extraction Complete')

            logging.info(f'Creating exceptions for endings with less than {count} words')
            idx_exceptions_1 = create_exception_1_index(count)
            logging.info(f'There are {len(idx_exceptions_1)} {count}-word endings that will not be altered due to number of words in string.')

            logging.info(f'Creating exceptions for endings that are next to prepositions.')
            idx_exceptions_2 = create_exception_2_index(count, old_column_name)
            logging.info(f'There are {len(idx_exceptions_2)} {count}-word endings that will not be altered due to proximity to a proposition.')
            
            logging.info(f'Creating exceptions for instances where remaining string length after ending removal is too small.')
            idx_exceptions_3 = create_exception_3_index(count, old_column_name, string_length=10)
            logging.info(f'There are {len(idx_exceptions_3)} {count}-word endings that will not be altered due to length of remaining string.')

            idx_exceptions = [*set(idx_exceptions_1 + idx_exceptions_2 + idx_exceptions_3)]
            idx_non_exceptions = self.df.index.difference(idx_exceptions).tolist()
            
            remove_common_endings(common_endings, idx_non_exceptions, old_column_name, new_column_name)
            populate_column_for_removed_endings(idx_non_exceptions, count, old_column_name, new_column_name)

        def create_ending_columns(ending_count):
            '''
            There is a method to the madness here. The main reason we're not just renaming the 
            n-word endings to remove, is because we want to see the individual strings that were removed.
            For example, if 'CONSTRUCTION INC' was a 2-word ending that was removed--we want to see
            the 'INC' and the 'CONSTRUCTION' removed separately.
            # NEED TO MAKE ENDING_COUNT GLOBAL OR SOMETHING HERE
            '''
            # Join the columns that were split to extract the endings
            # pdb.set_trace()
            columns_to_join = [x for x in self.df.columns if re.search('word_ending_removed', x)][::-1]
            self.df['removed_endings'] = self.df.apply(lambda row: ' '.join([str(row[col]) for col in columns_to_join if row[col] is not None]), axis=1)
            for count in range(0, ending_count + 1):
                new_column = f'{count}-ending'
                count = count + 1 #this it to account for the 0-indexing
                # Extract the nth word from the joined column
                self.df[new_column] = self.df['removed_endings'].apply(lambda x: x.split()[-count] if x and len(x.split()) >= count else np.nan)
            # pdb.set_trace()
            
        def filter_out_columns():
            self.df.rename(columns={self.cleaned_column_name + '_1' : self.column_name + '_CLN'}, inplace=True)
            columns_to_drop = [x for x in self.df.columns if 'removed' in x] + ['string_length'] # + ['removed_endings']
            more_columns_to_drop = [x for x in self.df.columns if 'cleaned' in x]
            self.df.drop(columns=columns_to_drop + more_columns_to_drop, inplace=True)
            # self.df.drop(columns=[x for x in self.df.columns if '-ending' in x], inplace=True)
            # pdb.set_trace()
        
        
        # Second pass for 1-word endings
        count = 1
        old_column_name = f'{self.cleaned_column_name}_{count}'
        new_column_name = f'{self.cleaned_column_name}_{count - 1}'
        # self.cleaned_column_name = old_column_name

        logging.info(f'Second pass for removing most common {count}-word endings')
        common_endings = get_most_common_endings(count, old_column_name)
        logging.info('Word Extraction Complete')
        
        # logging.info(f'Creating exceptions for endings with less than {count} words')
        # idx_exceptions_1 = create_exception_1_index(count)
        # logging.info(f'There are {len(idx_exceptions_1)} {count}-word endings that will not be altered due to number of words in string.')

        logging.info(f'Creating exceptions for endings that are next to prepositions.')
        idx_exceptions_2 = create_exception_2_index(count, old_column_name)
        logging.info(f'There are {len(idx_exceptions_2)} {count}-word endings that will not be altered due to proximity to a proposition.')
        
        logging.info(f'Creating exceptions for instances where remaining string length after ending removal is too small.')
        idx_exceptions_3 = create_exception_3_index(count, old_column_name, string_length=10)
        logging.info(f'There are {len(idx_exceptions_3)} {count}-word endings that will not be altered due to length of remaining string.')
        
        idx_exceptions = [*set(idx_exceptions_2 + idx_exceptions_3)]
        idx_non_exceptions = self.df.index.difference(idx_exceptions).tolist()
        
        remove_common_endings(common_endings, idx_non_exceptions, old_column_name, new_column_name)
        populate_column_for_removed_endings(idx_non_exceptions, 0, old_column_name, new_column_name)
        # pdb.set_trace()
        create_ending_columns(ending_count)
        filter_out_columns()
        
        return self.df


    def clean_entity_names(self) -> None:
        """Main method to clean company names."""
        # Perform initial cleaning steps
        # pdb.set_trace()
        self._perform_initial_cleaning()
        
        # Identify common endings in company names
        df = self._get_common_endings_and_remove(3, occurrence_count=20)
        return df