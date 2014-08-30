#include "mas_agent.h"
#include "worker_agent.h"
#include "constraint_store.h"
#include "aminoacid.h"
#include "utilities.h"
#include "search_engine.h"

using namespace std;

MasAgent::MasAgent ( MasAgentDes des, int prot_len ) :
_n_res           ( prot_len ),
_n_points        ( prot_len * 15 ),
_quantum         ( des.quantum ),
_priority        ( des.priority ),
_sum_dom_size    ( 0 ),
_max_dom_size    ( 0 ),
_end_search      ( false ),
_dbg             ( "#log: MasAgent - " ),
_agt_type        ( des.agt_type ),
_sec_str_type    ( des.sec_str_type ),
_search_strategy ( des.search_strategy ) {
  static int _g_MAS_COUNTER = 0;
  _id = _g_MAS_COUNTER++;
  
  _search_engine    = NULL;
  _current_status   = new real[ _n_points ];
  _constraint_store = new ConstraintStore ();
  
  _energy_weights[ 0 ] = _energy_weights[ 1 ] = _energy_weights[ 2 ] = 0;
  
  if ( des.vars_bds.size() > 0 )
    for (int i = 0; i < des.vars_bds.size(); i++)
      _vars_bds.push_back ( des.vars_bds[ i ] );
  
  _vars_list = des.vars_list;
  
  /// A sequence of amino acid here differs from the one represented in the AminoAcid class:
  /// N ( - Ca - C - O - H - N )^l - Ca - C - O - H
  /// where l+1 is the length of the piece of structure
  int min_aa = *std::min_element( _vars_list.begin(), _vars_list.end() );
  int max_aa = *std::max_element( _vars_list.begin(), _vars_list.end() );
  _atoms_bds.first  = Utilities::get_bbidx_from_aaidx ( min_aa, N );
  _atoms_bds.second = Utilities::get_bbidx_from_aaidx ( max_aa, H );
  
  if ( (des.scope.first == -1) && (_agt_type == coordinator) )
    _scope.first = 0;
  else if ( (des.scope.first == -1) && (_agt_type == structure) )
    _scope.first = min_aa;
  else if ( des.scope.first != -1 )
    _scope.first = des.scope.first;
  else
    _scope.first = 0;
  
  if ( (des.scope.second == -1) && (_agt_type == coordinator) )
    _scope.second = prot_len-1;
  else if ( (des.scope.second == -1) && (_agt_type == structure) )
    _scope.second = max_aa;
  else if ( des.scope.second != -1 )
    _scope.second = des.scope.second;
  else
    _scope.second = prot_len-1;
  /*
  /// Update min-max atoms refs considering the scope set by the user
  min_aa = min( min_aa, _scope.first );
  /// Handle case where the min aa is 0 and the coordinator is not the default one
  /// (necessary for merging structures)
  if ( (*std::min_element( _vars_list.begin(), _vars_list.end() )) > 0 ) { min_aa = max( min_aa, 1 ); }
  
  max_aa = max( max_aa, _scope.second );
  _atoms_bds.first  = Utilities::get_bbidx_from_aaidx ( min_aa, N );
  _atoms_bds.second = Utilities::get_bbidx_from_aaidx ( max_aa, H );
   */
}//-

MasAgent::~MasAgent() {
  if ( _vars_list.size() > 0 ) _vars_list.clear();
  if ( _vars_bds.size() > 0 ) _vars_bds.clear();
  if ( !_current_status )   delete [] _current_status;
  if ( !_search_engine )    delete _search_engine;
  if ( !_constraint_store ) delete _constraint_store;
}//-

MasAgent::MasAgent ( const MasAgent& other ) {
  _id               = other._id;
  _dbg              = other._dbg;
  _n_res            = other._n_res;
  _quantum          = other._quantum;
  _n_points         = other._n_points;
  _priority         = other._priority;
  _end_search       = other._end_search;
  _agt_type         = other._agt_type;
  _sec_str_type     = other._sec_str_type;
  _sum_dom_size     = other._sum_dom_size;
  _max_dom_size     = other._max_dom_size;
  _vars_list        = other._vars_list;
  _vars_bds         = other._vars_bds;
  _scope.first      = other._scope.first;
  _scope.second     = other._scope.second;
  _atoms_bds.first  = other._atoms_bds.first;
  _atoms_bds.second = other._atoms_bds.second;
  _constraint_store = new ConstraintStore ();
  _current_status   = new real[ _n_points ];
}//-

MasAgent&
MasAgent::operator= ( const MasAgent& other ) {
  if (this != &other) {
    _id               = other._id;
    _dbg              = other._dbg;
    _n_res            = other._n_res;
    _quantum          = other._quantum;
    _n_points         = other._n_points;
    _priority         = other._priority;
    _end_search       = other._end_search;
    _agt_type         = other._agt_type;
    _sec_str_type     = other._sec_str_type;
    _sum_dom_size     = other._sum_dom_size;
    _max_dom_size     = other._max_dom_size;
    _vars_list        = other._vars_list;
    _vars_bds         = other._vars_bds;
    _scope.first      = other._scope.first;
    _scope.second     = other._scope.second;
    _atoms_bds.first  = other._atoms_bds.first;
    _atoms_bds.second = other._atoms_bds.second;
    _constraint_store = new ConstraintStore ();
    _current_status   = new real[ _n_points ];
  }
  return *this;
}//-

void
MasAgent::end_search () {
  _end_search = true;
}//done

bool
MasAgent::done () const {
  return _end_search;
}//done

int
MasAgent::get_quantum () const {
  return _quantum;
}//get_quantum

void
MasAgent::set_quantum ( int q ) {
  _quantum = q;
}//set_quantum

int
MasAgent::get_priority () const {
  return _priority;
}//get_priority

void
MasAgent::inc_priority () {
  _priority++;
}//get_priority

void
MasAgent::set_energy_weight ( int en_field, real w ) {
  _energy_weights[ en_field ] = w;
}//set_energy_weight

real
MasAgent::get_energy_weight ( int en_field ) {
  return _energy_weights[ en_field ];
}//set_energy_weight

void
MasAgent::set_priority ( int p ) {
  _priority = p;
}//get_priority

int
MasAgent::get_n_res() const {
  return _n_res;
}//get_n_res

int
MasAgent::get_n_points() const {
  return _n_points;
}//get_n_points

agent_type
MasAgent::get_agt_type () const {
  return _agt_type;
}//get_agt_type

void
MasAgent::clear_data () {
  _vars_list.clear();
  _vars_bds.clear();
  _wrk_agt.clear();
}//clear_data

void
MasAgent::add_worker ( WorkerAgent* wrk ) {
  for (int i = 0; i < wrk->get_scope_size(); i++)
    _wrk_agt[ wrk->get_var_id( i ) ] = wrk;
  _sum_dom_size += wrk->get_dom_size();
  if ( wrk->get_dom_size() > _max_dom_size )
    _max_dom_size = wrk->get_dom_size();
}//add_worker

int
MasAgent::var_list_size () const {
  return _vars_list.size();
}//var_list_size

int
MasAgent::get_var_id ( int v ) const {
  if ( v >= _vars_list.size() ) return -1;
  return _vars_list[ v ];
}//var_list_size

int
MasAgent::get_bounds ( int b ) const {
  switch ( b ) {
    case 0:
      return _vars_bds[ 0 ].first;
    case 1:
      return _vars_bds[ _vars_bds.size() -1 ].second;
    default:
      return -1;
  }
}//get_bounds

int
MasAgent::get_atom_bounds ( int b ) const {
  switch ( b ) {
    case 0:
      return _atoms_bds.first;
    case 1:
      return _atoms_bds.second;
    default:
      return -1;
  }
}//get_atom_bounds

int
MasAgent::get_scope_start () const {
  return _scope.first;
}//get_scope_start

int
MasAgent::get_scope_end () const {
  return _scope.second;
}//get_scope_end

int
MasAgent::get_bb_start () const {
  return get_atom_bounds ( 0 );
}//get_bb_start

int
MasAgent::get_bb_end () const {
  return get_atom_bounds ( 1 );
}//get_bb_end

int
MasAgent::get_aa_start () const {
  return get_bounds ( 0 );
}//get_bb_start

int
MasAgent::get_aa_end () const {
  return get_bounds ( 1 );
}//get_bb_end

void
MasAgent::set_current_status ( real* c_status ) {
  memcpy ( _current_status, c_status, _n_points*sizeof( real ) );
  _search_engine->set_status( c_status, _n_points );
}//set_current_status

real*
MasAgent::get_current_status () {
  return _current_status;
}//get_current_status

map< int, WorkerAgent* >*
MasAgent::get_workers () {
  return &_wrk_agt;
}//get_workers

ConstraintStore*
MasAgent::get_c_store() {
  return _constraint_store;
}//get_c_store

void
MasAgent::dump_general_info () {
  cout << _dbg << "ID:\t" << _id << "\n";
  cout << _dbg << "priority:\t" << _priority << "\n";
  cout << _dbg << "end search:\t";
  ( _end_search ) ? cout << "yes\n" : cout << "no\n";
  cout << _dbg << "n. residues\t:" << _n_res << "\n";
  cout << _dbg << "n. points\t:" << _n_points << "\n";
  cout << _dbg << "max domains size:\t" << _max_dom_size << "\n";
  cout << _dbg << "sum domains size:\t" << _sum_dom_size << "\n";
}//dump_general_info

void
MasAgent::search_alloc ( int max_beam_size ) {
  Utilities::print_debug ( _dbg, "Search alloc" );
  gd_params.all_domains     = (real*) calloc ( (_n_res + (2 * _sum_dom_size)), sizeof(real) );
  gd_params.all_domains_idx = (int*) calloc ( _n_res,  sizeof(int) );
  gd_params.curr_str = (real*) calloc ( _n_points, sizeof(real) );
  gd_params.beam_str = (real*) calloc (   gh_params.set_size * _n_points, sizeof(real)  );
  gd_params.beam_energies = (real*) malloc( gh_params.set_size * sizeof(real) );
}//search_alloc

void
MasAgent::search_init () {
  Utilities::print_debug ( _dbg, "Search init" );
  real* all_doms       = (real*) calloc ( (_n_res + (2 * _sum_dom_size)), sizeof (real) );
  int* all_domains_idx = (int*)  calloc ( _n_res, sizeof (int) );
  
  WorkerAgent* curr_wrk;
  vector< pair < real, real >  >* domain;
  int z = 0, n_wrk = 0;
  for (map< int, WorkerAgent* >::iterator it=_wrk_agt.begin(); it!=_wrk_agt.end(); ++it) {
    curr_wrk = it->second;
    for (int i = 0; i < curr_wrk->get_scope_size(); i++) {
      domain = (curr_wrk->get_variable( i ))->get_domain_values();
      if ( n_wrk ) all_domains_idx[ (curr_wrk->get_variable( i ))->get_id() ] = z;
      all_doms[ z++ ] = domain->size();
      for (int j = 0; j < domain->size(); j++) {
        all_doms[ z++ ] = (domain->operator[](j)).first;
        all_doms[ z++ ] = (domain->operator[](j)).second;
      }
    }//i
    n_wrk++;
  }
  
  memcpy ( gd_params.all_domains, all_doms,
          (_n_res + (2 * _sum_dom_size)) * sizeof(real) );
  memcpy ( gd_params.all_domains_idx, all_domains_idx,
          _n_res * sizeof(int) );
  
  free ( all_doms );
  free ( all_domains_idx );
}//search_init

void
MasAgent::search_free() {
  Utilities::print_debug ( _dbg, "Search free" );
  free( gd_params.all_domains );
  free( gd_params.all_domains_idx );
  free( gd_params.curr_str );
  free( gd_params.beam_str );
  free( gd_params.beam_energies );
}//search_free


