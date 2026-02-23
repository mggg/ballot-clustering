# This file contains functions used for analysis of the IP and heuristic clustering results.

import numpy as np 
from Clustering_Functions import *
import itertools
from itertools import combinations

# Functions that reverse Borda and HH vectors, and sometimes yield generalized ballots.
def Borda_proxy(ballot, num_cands , borda_style='pes', start = 0):
    """
    Returns the Borda vector of the given (simple or generalized) ballot.
        
    Args:
        ballot : a simple or generalized ballot (tuple of integers or of sets of integers). 
        num_cands : the number of candidates. 
        borda_style : choice of {'pes', 'avg'}
        start : the lowest score awarded; for example, set start=1 if you want a full ballot to award {1,2,...,num_cands} points.
     
    Returns:
        the Borda vector (np.array) of the given generalized ballot.                
    """

    # append set of missing candidates to end of ballot
    ballot = list(ballot)
    missing_cands = set(range(1,num_cands+1))
    for c in ballot:
        S = c if type(c) == set else {c}
        for x in S:
            missing_cands.discard(x)
    if len(missing_cands) > 0:
        ballot.append(missing_cands)
    # compute Borda vector
    score_queue = list(range(start, start+num_cands))
    to_return = [0 for _ in range(num_cands)]
    for c in ballot:
        S = c if type(c) == set else {c}
        scores = [score_queue.pop() for _ in range(len(S))]
        points = np.mean(scores) if borda_style == 'avg' else min(scores)
        for x in S:
            to_return[x-1] = points

    return np.array(to_return)

def Reverse_Borda(proxy):
    """ 
    Returns the generalized ballot corresponding to the given Borda proxy vector.
    Returns a simple ballot if possible, otherwise a generalized ballot
    Works with either borda_style convention ('pes' or 'avg') 
    """
    num_cands = len(proxy)

    proxy = list(proxy)
    to_return = []
    cands_placed = []
    while len(cands_placed) < num_cands:
        S = [x for x in range(1,num_cands+1) if proxy[x-1]==np.max(proxy)] # best-scoring candidates
        cands_placed.extend(S)
        to_return.append(set(S))
        for x in S:
            proxy[x-1] = -1

    # return a simple ballot if possible
    if all(len(c)==1 for c in to_return[:-1]):
        return tuple([list(c)[0] for c in to_return if len(c)==1])
    else:
        return tuple(to_return)
    
def HH_proxy(ballot,num_cands):
    """
    Returns the head-to-head proxy vector of the given (simple or generalized) ballot.
        
    This is a vector with one entry for each pair of candidates ordered in the natural way; namely {(1,2),(1,3),...,(1,n),(2,3),...}. 
    The entries lie in {-1,0,1} depending on whether the lower-indexed candidate {looses, ties, wins} the head-to-head comparison. 

    Args:
        ballot: a simple or generalized ballot (tuple of integers or of sets of integers).
    
    Returns:
        The head-to-head proxy vector (np.array)
    """
    # append set of missing candidates to end of ballot
    ballot = list(ballot)
    missing_cands = set(range(1,num_cands+1))
    for c in ballot:
        S = c if type(c) == set else {c}
        for x in S:
            missing_cands.discard(x)
    if len(missing_cands) > 0:
        ballot.append(missing_cands)

    M = np.full([num_cands,num_cands], np.nan)

    # first place the zeros for ties and build the unpacked ballot
    unpacked_ballot = []
    for c in ballot:
        S = c if type(c) == set else {c}
        if len(S)>1:
            for x,y in combinations(S,2):
                M[x-1,y-1] = 0
                M[y-1,x-1] = 0
        unpacked_ballot.extend(S)

    # now place the -1 and 1 entries
    for x,y in combinations(unpacked_ballot,2):
        if M[x-1,y-1] != 0:
            M[x-1,y-1] = 1
            M[y-1,x-1] = -1

    # flatten the matrix into a vector
    to_return = []
    for x,y in combinations(range(num_cands),2):
        to_return.append(M[x,y])
    return np.array(to_return)

def Reverse_HH(proxy):
    """ 
    Returns the (simple or generalized) ballot corresponding to the given HH proxy vector,
    or None if the proxy is inconsistent.
    Any positive entry (not just +1) is interpreted as a win for the lower-indexed candidate, and any negative entry a loss,
    while a zero entry indicates a tie.
    Returns a simple ballot if possible, otherwise a generalized ballot.
    """
    # determine the number of candidates
    proxy = list(proxy)
    A = np.sqrt(1+8*len(proxy))
    if not A.is_integer():
        raise ValueError(f"Invalid proxy vector: {A, proxy}")
    num_cands = int((1+A)/2)
    
    cand_pairs = list(combinations(range(1,num_cands+1),2))
    ballot = [{num_cands}] # initialize ballot: bullet vote for last candidate

    # We'll work through cand_pairs (i,j) in reverse order.  
    # For 5 candidates, the order is (4,5) | (3,5), (3,4) | (2,5), (2,4), (2,3) | (1,5), (1,4), (1,3), (1,2)
    # which breaks into groups for i = 4,3,2,1
    # for each group, we add i to the top of the ballot, and then use the rest of the group's information to reposition i correctly
    # (or return None if the rest of the group has inconsisent information).
    for i in range(num_cands-1,0,-1):
        group_indices = [x for x in range(len(cand_pairs)) if cand_pairs[x][0]==i]
        left_of_i = [cand_pairs[x][1] for x in group_indices if proxy[x]<0]
        right_of_i = [cand_pairs[x][1] for x in group_indices if proxy[x]>0]
        match_i = [cand_pairs[x][1] for x in group_indices if proxy[x] == 0]
        ballot_map = [] # has one entry {-1,0,+1} for each set in the ballot, indicating whether the set should be left, right, or containing i.
        for c in ballot:
            S = c if type(c) == set else {c}
            if all(x in left_of_i for x in S):
                ballot_map.append(-1)
            elif all(x in right_of_i for x in S):
                ballot_map.append(1)
            elif all(x in match_i for x in S):
                ballot_map.append(0)
            else:
                return None # inconsistent proxy
            
        zero_indices = [x for x in range(len(ballot_map)) if ballot_map[x]==0]
        if (ballot_map != sorted(ballot_map)) or (len(zero_indices)>1):
            return None # inconsistent proxy
        
        if len(zero_indices)==0:
            insertion_index = len(ballot_map) if all(val <= 0 for val in ballot_map) else min([x for x in range(len(ballot_map)) if ballot_map[x] >=0])
            ballot.insert(insertion_index,{i})
        else:
            insertion_index = zero_indices[0]
            ballot[insertion_index] = ballot[insertion_index].union({i})
    # return a simple ballot if possible
    if all(len(c)==1 for c in ballot[:-1]):
        return tuple([list(c)[0] for c in ballot if len(c)==1])
    else:
        return tuple(ballot)
    
def is_simple(ballot):
    """
    Returns True if the given ballot is simple, False otherwise.
    """
    return all(type(c)== int for c in ballot)

# Helper functions for generating all possible generalized ballots
def set_partitions(items):
    """Yield all set partitions of items.
    Each partition is a list of sets.
    """
    items = list(items)
    if not items:
        yield []
        return

    first = items[0]
    for rest in set_partitions(items[1:]):

        # Option 1: put `first` into an existing block
        for i in range(len(rest)):
            new_part = [block.copy() for block in rest]
            new_part[i].add(first)
            yield new_part

        # Option 2: put `first` in its own block
        yield [{first}] + [block.copy() for block in rest]

# Helper functions for generating all possible generalized ballots
def ordered_set_partitions(items):
    """Yield all ordered set partitions of items.
    Each result is a list of sets.
    """
    for part in set_partitions(items):
        for ordering in itertools.permutations(part):
            # copy blocks to avoid shared references
            yield [block.copy() for block in ordering]

def all_possible_ballots(
    num_cands,
    include_empty_ballot=True,
    include_generalized_ballots=False,
    output_proxies=False
):
    """.
    Generate all possible ballots for the given number of candidates.
    Args:
        num_cands: number of candidates
        include_empty_ballot: if True, include the empty ballot
        include_generalized_ballots: if True, include generalized ballots (ordered set partitions); otherwise only include strict ballots (permutations)
        output_proxies: if False, yield ballots; if 'Borda' or 'HH', yield the corresponding proxy vectors
    Yields:
        ballots or their proxy vectors"""
    candidates = list(range(1, num_cands + 1))

    if include_generalized_ballots:
        # generalized ballots = ordered set partitions
        for ballot in ordered_set_partitions(candidates):
            if not ballot and not include_empty_ballot:
                continue

            if output_proxies is False:
                yield tuple(ballot)
            elif output_proxies == 'Borda':
                yield Borda_proxy(tuple(ballot), num_cands=num_cands, borda_style='pes')
            elif output_proxies == 'HH':
                yield HH_proxy(tuple(ballot), num_cands=num_cands)
            else:
                raise ValueError(f"Invalid value for output_proxies: {output_proxies}")

    else:
        # ordinary (strict) ballots
        start_length = 0 if include_empty_ballot else 1
        for length in range(start_length, num_cands + 1):
            for ballot in itertools.permutations(candidates, length):
                if output_proxies is False:
                    yield tuple(ballot)
                elif output_proxies == 'Borda':
                    yield Borda_proxy(tuple(ballot), num_cands=num_cands, borda_style='pes')
                elif output_proxies == 'HH':
                    yield HH_proxy(tuple(ballot), num_cands=num_cands)
                else:
                    raise ValueError(f"Invalid value for output_proxies: {output_proxies}")

# Define function to compute k=1 centers (the centers of the whole election)
def One_cluster_center(election, method = 'coords', proxy='Borda', borda_style='pes'):
    '''
    Returns the center of a single cluster using the specified method and proxy.
    if method = 'coords', return the L^1-center (the coordinate-wise median) (in proxy space).
    if method = 'cast', return the cast ballot that minimizes the total L1 distance to all other ballots (in ballot space).
    if method = 'Lloyd', return the L^2-center (the coordinate-wise mean) (in proxy space).
    if method = 'all', return the ballot (among all possible ballots) that minimizes the total L1 distance to all other ballots.
    '''
    num_cands = max([item for ranking in election.keys() for item in ranking])

    if method == 'coords' or method == 'Lloyd':
        # create a matrix whose rows are the proxies of the ballots (repeated as many times as the ballot was cast) 
        X = []
        for ballot, weight in election.items():
            for _ in range(weight):
                if proxy=='Borda':
                    X.append(Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style))
                else:
                    X.append(HH_proxy(ballot,num_cands=num_cands))
        X = np.array(X)
        # compute the coordinate-wise median or mean
        if method == 'coords':
            center_coords = np.median(X, axis=0)
        else:  # method == 'Lloyd'
            center_coords = np.mean(X, axis=0)
        return center_coords
    
    elif method == 'cast':
        min_total_distance = float('inf')
        best_ballot = None
        for ballot1 in election.keys():
            total_distance = 0
            for ballot2, weight in election.items():
                if proxy=='Borda':
                    dist = Borda_dist(ballot1, ballot2, num_cands=num_cands, borda_style=borda_style)
                else:
                    dist = HH_dist(ballot1, ballot2, num_cands=num_cands)
                total_distance += dist * weight
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_ballot = ballot1
        return best_ballot
    
    elif method == 'all':
        min_total_distance = float('inf')
        best_ballot = None
        for ballot1 in all_possible_ballots(num_cands, include_empty_ballot=True):
            total_distance = 0
            for ballot2, weight in election.items():
                if proxy=='Borda':
                    dist = Borda_dist(ballot1, ballot2, num_cands=num_cands, borda_style=borda_style)
                else:
                    dist = HH_dist(ballot1, ballot2, num_cands=num_cands)
                total_distance += dist * weight
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_ballot = ballot1
        return best_ballot

def Dist_to_closest_ballot(proxy_point, proxy_type='HH', include_empty_ballot=True, include_generalized_ballots=False, return_closest_ballots=False):
    """
    Given a point in proxy space, find the L^1-closest points in proxy space that correspond to actual ballots (or actual generalized ballots).
    Args:
        proxy_point: a point in proxy space
        proxy_type: 'Borda' or 'HH'
        include_generalized_ballots: if True, consider generalized ballots as well as simple ballots
        return_closest_ballots: if True, return also the closest ballots found.
    Returns:
        if return_closest_ballots is True:
            (list_of_closets_ballots, distance)
        else:
            distance
    """

    # find number of candidates
    proxy_len = len(proxy_point)
    if proxy_type == 'Borda':
        num_cands = proxy_len

    else:
        A = np.sqrt(1+8*proxy_len)
        if not A.is_integer():
            raise ValueError(f"Invalid proxy vector: {A, proxy_point}")
        num_cands = int((1+A)/2)
    
    closest_distance = float('inf')
    closest_ballots = []
    for ballot_proxy in all_possible_ballots(
        num_cands,
        include_empty_ballot=include_empty_ballot,
        include_generalized_ballots=include_generalized_ballots,
        output_proxies=proxy_type
    ):
        d = np.linalg.norm(np.array(proxy_point) - np.array(ballot_proxy), ord=1)
        if d < closest_distance:
            closest_distance = d
            if return_closest_ballots:
                closest_ballots = [Reverse_Borda(ballot_proxy) if proxy_type=='Borda' else Reverse_HH(ballot_proxy)]
        elif d == closest_distance:
            if return_closest_ballots:
                closest_ballots.append(Reverse_Borda(ballot_proxy) if proxy_type=='Borda' else Reverse_HH(ballot_proxy))
    if return_closest_ballots:
        return closest_ballots, closest_distance
    else:
        return closest_distance


# The following is an old version that I'm keeping for reference.
# It doesn't behave correctly when proxy_point has non-integer entries, which can happen with 'coords' centers
# because the median of integers need not be an integer.
def Find_closest_actual_ballot(proxy_point, proxy_type='HH', allow_generalized=False, return_only_one=False):
    """
    Given a point in proxy space, find the L^1-closest points in proxy space that correspond to actual ballots (or actual generalized ballots).
    Args:
        proxy_point: a point in proxy space
        proxy_type: 'Borda' or 'HH'
        allow_generalized: if True, consider generalized ballots as well as simple ballots
        return_only_one: if True, return only one of the closest ballots found, which speeds up the search.
    Returns:
        A list of ballots (in ballot space) that tie for closest, and the L^1 distance to those closest ballots.
    """

    # fund number of candidates and (inclusive) bounds for coordinates of valid ballots
    proxy_len = len(proxy_point)
    if proxy_type == 'Borda':
        num_cands = proxy_len
        lbound = 0
        ubound = num_cands - 1
    else:
        A = np.sqrt(1+8*proxy_len)
        if not A.is_integer():
            raise ValueError(f"Invalid proxy vector: {A, proxy_point}")
        num_cands = int((1+A)/2)
        lbound = -1
        ubound = 1
    R = -1
    closest_points = []
    while len(closest_points) == 0:
        R+=1
        # try all points with distance R from proxy_point
        for coords_to_change in itertools.product(range(proxy_len), repeat=R):
            coord_change_magnitude = {position:0 for position in range(proxy_len)}
            coords_that_change = []
            for coord in coords_to_change:
                coord_change_magnitude[coord] += 1
                if coord not in coords_that_change:
                    coords_that_change.append(coord)
            for signs_of_change in itertools.product([-1,1], repeat=len(coords_that_change)):
                sign_change_dict = {position:sign for position, sign in zip(coords_that_change, signs_of_change)}
                new_proxy = list(proxy_point)
                bad_point = False
                for coord in coords_that_change:
                    new_proxy[coord] += sign_change_dict[coord] * coord_change_magnitude[coord]
                    if new_proxy[coord] < lbound or new_proxy[coord] > ubound:
                        bad_point = True
                        break
                if bad_point:
                    continue
                # check if new_proxy corresponds to an actual ballot
                if proxy_type == 'Borda':
                    candidate_ballot = Reverse_Borda(new_proxy)
                else:
                    candidate_ballot = Reverse_HH(new_proxy)
                    if candidate_ballot == None:
                        continue
                if is_simple(candidate_ballot) or allow_generalized:
                    if proxy_type == 'Borda':
                        candidate_proxy = Borda_proxy(candidate_ballot, num_cands=num_cands, borda_style='pes') 
                    else:
                        candidate_proxy = HH_proxy(candidate_ballot, num_cands=num_cands)
                    if list(candidate_proxy) == list(new_proxy):
                        if candidate_ballot not in closest_points:
                            closest_points.append(candidate_ballot)
                            if return_only_one:
                                return closest_points, R
    return closest_points, R